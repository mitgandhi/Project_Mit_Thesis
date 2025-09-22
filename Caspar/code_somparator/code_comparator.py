import os
import difflib
import re
from pathlib import Path
from datetime import datetime


def get_cpp_files(folder_path):
    """Get all C++ files from a folder"""
    folder = Path(folder_path)
    extensions = ['.cpp', '.c', '.h', '.hpp', '.cc', '.cxx']

    files = {}
    for ext in extensions:
        for file_path in folder.rglob(f'*{ext}'):
            relative_path = file_path.relative_to(folder)
            files[str(relative_path)] = file_path

    return files


def read_file_safe(file_path):
    """Read file with encoding fallback"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.readlines()
    except Exception:
        return []


def extract_functions_and_classes(lines):
    """Extract functions and classes with their line ranges"""
    functions = []
    current_function = None
    brace_count = 0
    in_function = False

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Skip empty lines and comments for function detection
        if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
            continue

        # Detect function/class/struct definitions
        # Look for patterns like: return_type function_name(...) or class/struct name
        function_patterns = [
            r'^\s*(?:(?:inline|static|virtual|extern|friend|explicit|const|constexpr|noexcept|override|final)\s+)*\s*(?:(?:unsigned|signed|long|short|const|static|volatile|mutable|register|auto|extern|thread_local)\s+)*(?:[a-zA-Z_][a-zA-Z0-9_:]*(?:\s*<[^>]*>)?\s*[&*]*\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^{;]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?(?:noexcept\s*)?(?:\s*->\s*[^{;]+)?\s*\{',
            r'^\s*(?:class|struct|namespace)\s+([a-zA-Z_][a-zA-Z0-9_]*)[^{;]*\{',
            r'^\s*(?:enum\s+(?:class\s+)?)\s*([a-zA-Z_][a-zA-Z0-9_]*)[^{;]*\{'
        ]

        for pattern in function_patterns:
            match = re.search(pattern, line)
            if match:
                if current_function and not in_function:
                    # End previous function if we weren't tracking braces
                    current_function['end_line'] = i - 1
                    functions.append(current_function)

                current_function = {
                    'name': match.group(1),
                    'start_line': i,
                    'end_line': None,
                    'type': 'class' if 'class' in line else 'struct' if 'struct' in line else 'namespace' if 'namespace' in line else 'enum' if 'enum' in line else 'function'
                }
                brace_count = line.count('{') - line.count('}')
                in_function = True
                break

        if in_function:
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                current_function['end_line'] = i
                functions.append(current_function)
                current_function = None
                in_function = False

    # Handle case where file ends while in a function
    if current_function:
        current_function['end_line'] = len(lines) - 1
        functions.append(current_function)

    return functions


def get_line_changes(diff_lines):
    """Extract which lines were added, removed, or modified with their content"""
    changes = {'added': {}, 'removed': {}, 'modified': {}}

    current_old_line = 0
    current_new_line = 0

    for line in diff_lines:
        if line.startswith('@@'):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if match:
                current_old_line = int(match.group(1)) - 1
                current_new_line = int(match.group(3)) - 1
        elif line.startswith('-') and not line.startswith('---'):
            changes['removed'][current_old_line] = line[1:]
            current_old_line += 1
        elif line.startswith('+') and not line.startswith('+++'):
            changes['added'][current_new_line] = line[1:]
            current_new_line += 1
        elif not line.startswith(('---', '+++')):
            # Context line
            current_old_line += 1
            current_new_line += 1

    return changes


def get_changed_sections(old_lines, new_lines, diff_lines):
    """Extract separate added and removed sections with context"""
    changes = {'added_sections': [], 'removed_sections': []}

    current_old_line = 0
    current_new_line = 0

    # Parse diff to find change hunks
    i = 0
    while i < len(diff_lines):
        line = diff_lines[i]

        if line.startswith('@@'):
            # Parse hunk header
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if match:
                old_start = int(match.group(1)) - 1
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3)) - 1
                new_count = int(match.group(4)) if match.group(4) else 1

                # Extract the hunk content
                hunk_lines = []
                i += 1
                while i < len(diff_lines) and not diff_lines[i].startswith('@@'):
                    hunk_lines.append(diff_lines[i])
                    i += 1
                i -= 1  # Back up one since the outer loop will increment

                # Process this hunk to separate added and removed sections
                removed_lines = []
                added_lines = []
                context_before = []
                context_after = []

                # Collect context before changes
                for line in hunk_lines:
                    if line.startswith(' ') and not removed_lines and not added_lines:
                        context_before.append(line[1:])
                    elif line.startswith('-'):
                        removed_lines.append(line[1:])
                    elif line.startswith('+'):
                        added_lines.append(line[1:])
                    elif line.startswith(' ') and (removed_lines or added_lines):
                        context_after.append(line[1:])
                        break

                # Create removed section if there are removals
                if removed_lines:
                    section = {
                        'start_line': old_start,
                        'context_before': context_before[-3:],  # Last 3 lines of context
                        'content': removed_lines,
                        'context_after': context_after[:3],  # First 3 lines of context
                        'type': 'removed'
                    }
                    changes['removed_sections'].append(section)

                # Create added section if there are additions
                if added_lines:
                    section = {
                        'start_line': new_start,
                        'context_before': context_before[-3:],  # Last 3 lines of context
                        'content': added_lines,
                        'context_after': context_after[:3],  # First 3 lines of context
                        'type': 'added'
                    }
                    changes['added_sections'].append(section)

        i += 1

    return changes


def count_meaningful_changes(diff_lines):
    """Count only meaningful changes (ignore whitespace-only changes)"""
    additions = 0
    deletions = 0

    for line in diff_lines:
        if line.startswith('+') and not line.startswith('+++'):
            content = line[1:].strip()
            if content:  # Not just whitespace
                additions += 1
        elif line.startswith('-') and not line.startswith('---'):
            content = line[1:].strip()
            if content:  # Not just whitespace
                deletions += 1

    return additions, deletions


def get_change_summary(additions, deletions):
    """Get a simple change summary with clearer explanations"""
    if additions == 0 and deletions == 0:
        return "🟢 No meaningful changes detected"
    elif additions > 0 and deletions == 0:
        return f"🟡 {additions} lines of code were added"
    elif additions == 0 and deletions > 0:
        return f"🔴 {deletions} lines of code were removed"
    else:
        net_change = additions - deletions
        if net_change > 0:
            return f"🔵 {additions} additions and {deletions} deletions (net growth of {net_change} lines)"
        elif net_change < 0:
            return f"🔵 {additions} additions and {deletions} deletions (net reduction of {abs(net_change)} lines)"
        else:
            return f"🔵 {additions} additions and {deletions} deletions (no net change in line count)"


def get_change_type(diff_content):
    """Identify what type of changes were made"""
    changes = []
    content = ' '.join(diff_content).lower()

    if 'class ' in content or 'struct ' in content:
        changes.append("Class/Struct definitions")
    if 'function' in content or '(' in content and ')' in content:
        changes.append("Functions")
    if '#include' in content:
        changes.append("Include statements")
    if '//' in content or '/*' in content:
        changes.append("Comments")
    if 'if(' in content or 'for(' in content or 'while(' in content:
        changes.append("Control flow")
    if '=' in content and not '==' in content:
        changes.append("Variable assignments")

    return changes if changes else ["Code logic"]


def compare_folders_detailed(new_folder, old_folder, output_file):
    """Detailed file-by-file comparison with complete functions and color coding"""

    new_files = get_cpp_files(new_folder)
    old_files = get_cpp_files(old_folder)
    all_files = set(new_files.keys()) | set(old_files.keys())

    # Categorize files
    file_status = {}
    total_additions = 0
    total_deletions = 0

    for file_name in sorted(all_files):
        status = {}

        if file_name not in new_files:
            status['type'] = 'REMOVED'
            status['summary'] = "🗑️ File removed in v_3.2.0"
            status['size_old'] = old_files[file_name].stat().st_size

        elif file_name not in old_files:
            status['type'] = 'NEW'
            status['summary'] = "✨ New file in v_3.2.0"
            status['size_new'] = new_files[file_name].stat().st_size

        else:
            # Compare existing files
            new_content = read_file_safe(new_files[file_name])
            old_content = read_file_safe(old_files[file_name])

            if new_content == old_content:
                status['type'] = 'IDENTICAL'
                status['summary'] = "✅ No changes"
            else:
                status['type'] = 'MODIFIED'

                # Analyze changes
                diff = list(difflib.unified_diff(old_content, new_content, lineterm=""))
                additions, deletions = count_meaningful_changes(diff)

                status['additions'] = additions
                status['deletions'] = deletions
                status['summary'] = get_change_summary(additions, deletions)
                status['change_types'] = get_change_type([line for line in diff if line.startswith(('+', '-'))])
                status['diff'] = diff
                status['old_content'] = old_content
                status['new_content'] = new_content

                # Extract functions from both versions
                status['old_functions'] = extract_functions_and_classes(old_content)
                status['new_functions'] = extract_functions_and_classes(new_content)

                # Get separate added and removed sections
                status['changed_sections'] = get_changed_sections(old_content, new_content, diff)

                total_additions += additions
                total_deletions += deletions

            status['size_old'] = old_files[file_name].stat().st_size if file_name in old_files else 0
            status['size_new'] = new_files[file_name].stat().st_size if file_name in new_files else 0

        file_status[file_name] = status

    # Generate HTML report for better color coding
    with open(output_file.replace('.md', '.html'), 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Code Comparison Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .summary-table th, .summary-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        .summary-table th { background-color: #f2f2f2; font-weight: bold; }
        .function-container { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; margin: 10px 0; padding: 15px; }
        .function-header { background: #343a40; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: monospace; font-weight: bold; }
        .code-block { background: #ffffff; border: 1px solid #dee2e6; border-radius: 5px; padding: 15px; margin: 10px 0; font-family: 'Courier New', monospace; white-space: pre-wrap; line-height: 1.4; }
        .added-line { background-color: #d4edda !important; color: #155724 !important; display: block; margin: 2px 0; padding: 2px 5px; border-left: 4px solid #28a745; }
        .removed-line { background-color: #f8d7da !important; color: #721c24 !important; display: block; margin: 2px 0; padding: 2px 5px; border-left: 4px solid #dc3545; }
        .unchanged-line { display: block; margin: 1px 0; padding: 2px 5px; }
        .section { margin: 30px 0; padding: 20px; border-radius: 8px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .file-title { color: #495057; border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-bottom: 20px; }
        .legend { background: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .legend-item { margin: 5px 0; }
        .no-changes { color: #6c757d; font-style: italic; }
    </style>
</head>
<body>
""")

        f.write(f"""
        <div class="header">
            <h1>📊 Code Comparison Report - Complete Functions View</h1>
            <p><strong>📁 Version v_3.2.0 (New):</strong> {new_folder}</p>
            <p><strong>📁 Version v_2.0.0 (Old):</strong> {old_folder}</p>
            <p><strong>📅 Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """)

        # Quick Summary
        modified = [f for f, s in file_status.items() if s['type'] == 'MODIFIED']
        new_files_list = [f for f, s in file_status.items() if s['type'] == 'NEW']
        removed_files = [f for f, s in file_status.items() if s['type'] == 'REMOVED']
        identical = [f for f, s in file_status.items() if s['type'] == 'IDENTICAL']

        f.write("""
        <div class="section">
            <h2>🎯 Quick Summary</h2>
            <table class="summary-table">
                <tr><th>Status</th><th>Count</th><th>Files</th></tr>
        """)

        f.write(
            f"<tr><td>🔵 <strong>Modified</strong></td><td>{len(modified)}</td><td>{', '.join(modified[:3])}{'...' if len(modified) > 3 else ''}</td></tr>")
        f.write(
            f"<tr><td>✨ <strong>New in v_3.2.0</strong></td><td>{len(new_files_list)}</td><td>{', '.join(new_files_list[:3])}{'...' if len(new_files_list) > 3 else ''}</td></tr>")
        f.write(
            f"<tr><td>🗑️ <strong>Removed from v_2.0.0</strong></td><td>{len(removed_files)}</td><td>{', '.join(removed_files[:3])}{'...' if len(removed_files) > 3 else ''}</td></tr>")
        f.write(f"<tr><td>✅ <strong>Identical</strong></td><td>{len(identical)}</td><td>No changes</td></tr>")
        f.write(
            f"<tr><td><strong>📊 Total Changes</strong></td><td><strong>+{total_additions} -{total_deletions}</strong></td><td><strong>Net: {total_additions - total_deletions:+d} lines</strong></td></tr>")

        f.write("""
            </table>
        </div>
        """)

        # Legend
        f.write("""
        <div class="legend">
            <h3>🔹 How to Read the Code Changes:</h3>
            <div class="legend-item"><span class="removed-line" style="display: inline; padding: 2px 5px;">🗑️ Red sections</span> show code that was REMOVED from v_2.0.0</div>
            <div class="legend-item"><span class="added-line" style="display: inline; padding: 2px 5px;">✨ Green sections</span> show code that was ADDED in v_3.2.0</div>
            <div class="legend-item"><span class="unchanged-line" style="display: inline; padding: 2px 5px;">📄 Gray lines</span> provide context around changes</div>
            <p><strong>New Approach:</strong> Instead of mixing added/removed lines together, each change is shown as a separate section with clear context, making it easier to understand what was removed vs what was added.</p>
        </div>
        """)

        # File-by-File Analysis with Complete Functions
        f.write('<div class="section"><h2>📋 File-by-File Analysis - Complete Functions</h2>')

        # Process modified files with complete functions
        for file_name in sorted(modified):
            status = file_status[file_name]

            f.write(f'<div class="section">')
            f.write(f'<h3 class="file-title">📄 {file_name}</h3>')
            f.write(f'<p><strong>Status:</strong> {status["summary"]}</p>')

            # File size info
            size_diff = status['size_new'] - status['size_old']
            f.write(f'<p><strong>Size:</strong> {status["size_old"]:,} → {status["size_new"]:,} bytes ')
            if size_diff != 0:
                f.write(f'({size_diff:+,} bytes)</p>')
            else:
                f.write('(same size)</p>')

            f.write(
                f'<p><strong>Changes:</strong> +{status["additions"]} lines added, -{status["deletions"]} lines removed</p>')

            if status['change_types']:
                f.write(f'<p><strong>Types of Changes:</strong> {", ".join(status["change_types"])}</p>')

            # Show changes as separate sections
            f.write('<h4>🔧 Code Changes (Separated by Type):</h4>')

            changed_sections = status.get('changed_sections', {'removed_sections': [], 'added_sections': []})

            if not changed_sections['removed_sections'] and not changed_sections['added_sections']:
                f.write('<p class="no-changes">No significant code sections with changes detected.</p>')
            else:
                # Show removed sections first
                if changed_sections['removed_sections']:
                    f.write('<h5>🗑️ Removed Code Sections:</h5>')
                    for i, section in enumerate(changed_sections['removed_sections']):
                        f.write(f'<div class="function-container">')
                        f.write(
                            f'<div class="function-header">❌ Removed Section {i + 1} (around line {section["start_line"] + 1})</div>')
                        f.write(f'<div class="code-block">')

                        # Show context before
                        if section['context_before']:
                            f.write(
                                '<span style="color: #6c757d; font-style: italic;">// ... context before ...</span>\n')
                            for line in section['context_before']:
                                line_content = line.rstrip('\n').replace('<', '&lt;').replace('>', '&gt;').replace('&',
                                                                                                                   '&amp;')
                                f.write(f'<span class="unchanged-line">  {line_content}</span>')

                        # Show removed content
                        for line in section['content']:
                            line_content = line.rstrip('\n').replace('<', '&lt;').replace('>', '&gt;').replace('&',
                                                                                                               '&amp;')
                            f.write(f'<span class="removed-line">- {line_content}</span>')

                        # Show context after
                        if section['context_after']:
                            for line in section['context_after']:
                                line_content = line.rstrip('\n').replace('<', '&lt;').replace('>', '&gt;').replace('&',
                                                                                                                   '&amp;')
                                f.write(f'<span class="unchanged-line">  {line_content}</span>')
                            f.write('<span style="color: #6c757d; font-style: italic;">// ... context after ...</span>')

                        f.write(f'</div></div>')

                # Show added sections
                if changed_sections['added_sections']:
                    f.write('<h5>✨ Added Code Sections:</h5>')
                    for i, section in enumerate(changed_sections['added_sections']):
                        f.write(f'<div class="function-container">')
                        f.write(
                            f'<div class="function-header">✅ Added Section {i + 1} (around line {section["start_line"] + 1})</div>')
                        f.write(f'<div class="code-block">')

                        # Show context before
                        if section['context_before']:
                            f.write(
                                '<span style="color: #6c757d; font-style: italic;">// ... context before ...</span>\n')
                            for line in section['context_before']:
                                line_content = line.rstrip('\n').replace('<', '&lt;').replace('>', '&gt;').replace('&',
                                                                                                                   '&amp;')
                                f.write(f'<span class="unchanged-line">  {line_content}</span>')

                        # Show added content
                        for line in section['content']:
                            line_content = line.rstrip('\n').replace('<', '&lt;').replace('>', '&gt;').replace('&',
                                                                                                               '&amp;')
                            f.write(f'<span class="added-line">+ {line_content}</span>')

                        # Show context after
                        if section['context_after']:
                            for line in section['context_after']:
                                line_content = line.rstrip('\n').replace('<', '&lt;').replace('>', '&gt;').replace('&',
                                                                                                                   '&amp;')
                                f.write(f'<span class="unchanged-line">  {line_content}</span>')
                            f.write('<span style="color: #6c757d; font-style: italic;">// ... context after ...</span>')

                        f.write(f'</div></div>')

            f.write('</div>')

        # Process other file types (new, removed, identical) - simplified view
        for status_type, emoji, title in [
            ('NEW', '✨', 'New Files in v_3.2.0'),
            ('REMOVED', '🗑️', 'Files Removed from v_2.0.0'),
            ('IDENTICAL', '✅', 'Identical Files (No Changes)')
        ]:
            files_of_type = [f for f, s in file_status.items() if s['type'] == status_type]
            if files_of_type:
                f.write(f'<div class="section">')
                f.write(f'<h3>{emoji} {title} ({len(files_of_type)} files)</h3>')
                f.write('<ul>')
                for file_name in sorted(files_of_type):
                    status = file_status[file_name]
                    f.write(f'<li><strong>{file_name}</strong> - {status["summary"]}')
                    if 'size_new' in status:
                        f.write(f' ({status["size_new"]:,} bytes)')
                    elif 'size_old' in status:
                        f.write(f' ({status["size_old"]:,} bytes)')
                    f.write('</li>')
                f.write('</ul>')
                f.write('</div>')

        # Final recommendations
        f.write('<div class="section">')
        f.write('<h2>💡 Recommendations</h2>')

        if len(modified) == 0 and len(new_files_list) == 0 and len(removed_files) == 0:
            f.write('<p>✅ No action needed - versions are identical</p>')
        else:
            if len(modified) > 0:
                f.write(f'<p>🔍 <strong>Review {len(modified)} modified files</strong> for logic changes</p>')
            if len(new_files_list) > 0:
                f.write(f'<p>📝 <strong>Check {len(new_files_list)} new files</strong> for integration requirements</p>')
            if len(removed_files) > 0:
                f.write(
                    f'<p>⚠️ <strong>Verify {len(removed_files)} removed files</strong> don\'t break dependencies</p>')
            if total_additions > total_deletions * 2:
                f.write('<p>📈 <strong>Significant code growth</strong> - consider additional testing</p>')

        f.write('</div>')
        f.write('<hr><p><em>Report generated by Enhanced Code Comparison Tool with Complete Function View</em></p>')
        f.write('</body></html>')


def main():
    """Main function - UPDATE THESE PATHS"""

    # 🔧 CONFIGURE YOUR PATHS HERE:
    new_folder = 'E:/Job_1/SHS--version_3.1.3/FSTI/Piston/Caspar/mesh/src'
    old_folder = 'E:/Job_1/SHS--Original/localCopy/FSTI/piston/Caspar/mesh/src'
    output_file = 'detailed_code_comparison.html'

    print("🔍 Enhanced Version Comparison Tool (v_2.0.0 → v_3.2.0)")
    print("=" * 60)
    print("✨ NEW FEATURES:")
    print("   • Separate sections for REMOVED vs ADDED code")
    print("   • Context lines around each change")
    print("   • Clear visual separation of change types")
    print("   • HTML output for better formatting")
    print("   • No more mixed diff confusion!")
    print("=" * 60)

    # Validate paths
    if not os.path.exists(new_folder):
        print(f"❌ Error: Version v_3.2.0 folder '{new_folder}' not found!")
        print("💡 Please update the 'new_folder' path in the script")
        return

    if not os.path.exists(old_folder):
        print(f"❌ Error: Version v_2.0.0 folder '{old_folder}' not found!")
        print("💡 Please update the 'old_folder' path in the script")
        return

    print(f"📁 Comparing versions:")
    print(f"   🆕 Version v_3.2.0: {new_folder}")
    print(f"   📁 Version v_2.0.0: {old_folder}")
    print(f"   📄 Output: {output_file}")
    print()

    try:
        start_time = datetime.now()
        compare_folders_detailed(new_folder, old_folder, output_file)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()
        print(f"✅ Analysis complete in {duration:.1f} seconds!")
        print(f"📊 Enhanced HTML report saved to: '{output_file}'")
        print()
        print("🎯 Enhanced Report includes:")
        print("   • Separate sections for removed vs added code")
        print("   • Context lines around each change")
        print("   • Clear visual distinction between change types")
        print("   • Red sections for removed code")
        print("   • Green sections for added code")
        print("   • Beautiful HTML formatting")
        print()
        print("💡 Open the HTML file in your browser for the best viewing experience!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# Updated matplotlib parameters for professional styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.figsize': (8, 8),
    'axes.linewidth': 1.5
})

# Define colors for consistent styling
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Load data
caspar_gamma_0 = pd.read_csv('Caspar_gamma_0.csv')
caspar_gamma_5 = pd.read_csv('Caspar_gamma_5.csv')
gamma_0_data = pd.read_csv('python_forces_gamma_0.csv')
gamma_2_data = pd.read_csv('python_forces_gamma_2.csv')
gamma_4_data = pd.read_csv('python_forces_gamma_5.csv')
gamma_6_data = pd.read_csv('python_forces_gamma_6.csv')
gamma_8_data = pd.read_csv('python_forces_gamma_8.csv')
gamma_10_data = pd.read_csv('python_forces_gamma_10.csv')


def plt_FSK():
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')

    # Define gamma angles and corresponding data
    gamma_data = [
        (0, gamma_0_data),
        (5, gamma_4_data),
        (10, gamma_10_data)
    ]

    # Plot data with consistent colors and styling
    for i, (gamma, data) in enumerate(gamma_data):
        color = colors[i % len(colors)]
        ax.plot(data['shaft_angle'],
                data['FSKy'] / np.sin(np.radians(14)),
                linewidth=3,
                linestyle='--',
                color=color,
                label=f'γ = {gamma}°')

    # Styling
    ax.set_xlabel('Shaft Angle [deg]', fontsize=20, fontweight='bold')
    ax.set_ylabel('Swashplate-Reaction Force [N]', fontsize=20, fontweight='bold')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add grid with subtle styling
    ax.grid(True, alpha=0.4, linewidth=1, color='gray')

    # Create legend
    legend = ax.legend(fontsize=18,
                       loc='upper right',
                       frameon=True,
                       fancybox=True,
                       shadow=False,
                       framealpha=0.8,
                       edgecolor='black')
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    plt.savefig('FSK.png', dpi=300, bbox_inches='tight',
                transparent=True, facecolor='none', edgecolor='none')
    plt.show()


def plt_FSKy():
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')

    # Define gamma angles and corresponding data
    gamma_data = [
        (0, gamma_0_data),
        (5, gamma_4_data),
        (10, gamma_10_data)
    ]

    # Plot data with consistent colors and styling
    for i, (gamma, data) in enumerate(gamma_data):
        color = colors[i % len(colors)]
        ax.plot(data['shaft_angle'],
                data['FSKy'] + data['FAKy'],
                linewidth=3,
                linestyle='--',
                color=color,
                label=f'γ = {gamma}°')

    # Styling
    ax.set_xlabel('Shaft Angle [deg]', fontsize=20, fontweight='bold')
    ax.set_ylabel('Lateral Force [N]', fontsize=20, fontweight='bold')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add grid with subtle styling
    ax.grid(True, alpha=0.4, linewidth=1, color='gray')

    # Create legend
    legend = ax.legend(fontsize=18,
                       loc='upper right',
                       frameon=True,
                       fancybox=True,
                       shadow=False,
                       framealpha=0.8,
                       edgecolor='black')
    legend.get_frame().set_facecolor('white')

    plt.tight_layout()
    plt.savefig('FSKY.png', dpi=300, bbox_inches='tight',
                transparent=True, facecolor='none', edgecolor='none')
    plt.show()


def plt_FSK_combined():
    """Combined plot showing both research and literature data like the reference style"""
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')

    # Define gamma angles
    gamma_angles = [0, 5, 10]
    gamma_data_mapping = {
        0: gamma_0_data,
        5: gamma_4_data,
        10: gamma_10_data
    }

    # Plot with different line styles for different data sources
    for i, gamma in enumerate(gamma_angles):
        color = colors[i % len(colors)]
        data = gamma_data_mapping[gamma]

        # Plot as "research" data (solid lines)
        ax.plot(data['shaft_angle'],
                data['FSKy'] / np.sin(np.radians(14)),
                linewidth=3,
                linestyle='-',
                color=color,
                label=f'Research γ = {gamma}°')

        # If you have literature data, uncomment and modify this section:
        # ax.plot(data['shaft_angle'],
        #         literature_data_column,
        #         linewidth=3,
        #         linestyle='--',
        #         color=color,
        #         label=f'Literature γ = {gamma}°')

    # Styling
    ax.set_xlabel('Shaft Angle [deg]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Swashplate-Reaction Force [N]', fontsize=16, fontweight='bold')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add grid with subtle styling
    ax.grid(True, alpha=0.4, linewidth=1, color='gray')

    # Create custom legend for line styles
    custom_lines = [Line2D([0], [0], color='black', lw=3, linestyle='-'),
                    Line2D([0], [0], color='black', lw=3, linestyle='--')]
    legend = ax.legend(custom_lines, ['Research', 'Literature'],
                       fontsize=18,
                       loc='upper center',
                       ncol=2,
                       frameon=True,
                       fancybox=True,
                       shadow=False,
                       framealpha=0.8,
                       edgecolor='black')
    legend.get_frame().set_facecolor('white')

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig('FSK_combined.png', dpi=300, bbox_inches='tight',
                transparent=True, facecolor='none', edgecolor='none')
    plt.show()


# Run the plotting functions
if __name__ == "__main__":
    plt_FSKy()
    plt_FSK()
    plt_FSK_combined()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.figsize': (10, 6),
    'font.family': 'serif'
})


def read_data_file(filename):
    try:
        df = pd.read_csv(filename)
        if len(df.columns) == 1:
            for delimiter in ['\t', ' ', ';', '|']:
                try:
                    df = pd.read_csv(filename, delimiter=delimiter)
                    if len(df.columns) > 1:
                        break
                except:
                    continue
        df = df.dropna(axis=1, how='all')
        df = df.dropna(how='all')
        return df
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        return None


def read_piston_data(file_path):
    try:
        df = read_data_file(file_path)
        if df is None:
            return None, None, None

        required_cols = ['Total_Mechanical_Power_Loss', 'Total_Volumetric_Power_Loss']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Missing columns in {file_path}: {missing_cols}")
            return None, None, None

        mech_power_loss_mean = df['Total_Mechanical_Power_Loss'].mean()
        vol_power_loss_mean = df['Total_Volumetric_Power_Loss'].mean()
        total_power_loss = mech_power_loss_mean + vol_power_loss_mean

        return mech_power_loss_mean, vol_power_loss_mean, total_power_loss
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None


def process_single_folder(folder_path):
    folder_data = {}

    if not os.path.exists(folder_path):
        print(f"Error: Path does not exist: {folder_path}")
        return {}

    folder_items = os.listdir(folder_path)
    speed_folders = []

    for item in folder_items:
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and item.isdigit():
            speed_folders.append(int(item))

    speed_folders.sort()
    print(f"Found speed folders: {speed_folders}")

    for speed_val in speed_folders:
        speed_folder_path = os.path.join(folder_path, str(speed_val))
        piston_file = os.path.join(speed_folder_path, 'output', 'piston', 'piston.txt')

        if os.path.exists(piston_file):
            mech_loss, vol_loss, total_loss = read_piston_data(piston_file)

            if mech_loss is not None and vol_loss is not None:
                folder_data[speed_val] = {
                    'mechanical_loss': mech_loss,
                    'volumetric_loss': vol_loss,
                    'total_loss': total_loss,
                    'speed': speed_val
                }
                print(f"Speed {speed_val}: Total={total_loss:.2f}W")

    return folder_data


def create_power_loss_vs_speed_plot(folder_data, folder_name="Simulation"):
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.figsize': (10, 8),
        'font.family': 'serif'
    })

    if not folder_data:
        print("No data available for plotting")
        return None, None

    speeds = sorted(list(folder_data.keys()))
    total_losses = [folder_data[speed]['total_loss'] for speed in speeds]

    fig, ax = plt.subplots(figsize=(8, 8))

    x = np.arange(len(speeds))
    width = 0.6

    bars = ax.bar(x, total_losses, width, color='#2E86AB', alpha=0.8,
                  label='Total Power Loss', edgecolor='black', linewidth=1)

    for bar, total in zip(bars, total_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(total_losses) * 0.01,
                f'{total:.0f}W', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_xlabel('Speed (RPM)', fontweight='bold')
    ax.set_ylabel('Power Loss (W)', fontweight='bold')
    ax.set_title(f'Power Loss vs Speed\n(Pressure: 320bar, Displacement: 100%)',
                 fontweight='bold', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels([str(speed) for speed in speeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filename = f'power_loss_vs_speed_{folder_name.replace(" ", "_").replace("-", "_")}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")

    return fig, ax


if __name__ == "__main__":
    folder_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T05_variable speed_ speedK_1"
    folder_name = "sample"

    print(f"Processing folder: {folder_path}")
    folder_data = process_single_folder(folder_path)

    if folder_data:
        fig, ax = create_power_loss_vs_speed_plot(folder_data, folder_name)
        if fig is not None:
            plt.show()
    else:
        print("No data found to plot")
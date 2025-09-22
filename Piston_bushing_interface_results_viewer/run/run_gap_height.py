
# run_gap_height.py
from Gap_height import piston_gap_height

filepaths = [
    # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run1_Test_V60N\Run_V60N\V60N_S4_dp_350_n_4400_d_100',
    # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run2_Test_inclined_&_inclined_code-V60N\Run_inclined_V60N\V60N_S4_HP_dp_380_n_4400_d_100'
    # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_L62.4_Without_energy_equation\T01_4400n_380dp_100d',
    # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run2_L_57_without_energy_equation\T01_4400n_380dp_100d',
    # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run2_short_piston_bushing_without_energy_equation\T01_4100n_380dp_100d',
    # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run2_short_piston_bushing_without_energy_equation\T01_3800n_380dp_100d'
    # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run4_L_62.4_D_19.4\T01_4400n_380dp_100d'
    r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP_FD\Run\Run1\T01_4400n_380dp_100d',
    r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T01_4400n_380dp_100d'
]

labels = ['4400 rpm', '4400 rpm']
m = [80, 80]
lF = [39.5, 39.5]
n = [4400, 4400]
deltap = [380, 380]
offset = [0, 0]

plots = 360
degstep = 1
ignore_base = 0
minplot = 0
maxplot = 0

for i in range(len(filepaths)):
    print(f"\n=== Case {i+1}: {labels[i]} ===")
    piston_gap_height(
        filepath=filepaths[i],
        m=m[i],
        lF=lF[i],
        n=n[i],
        deltap=deltap[i],
        plots=plots,
        degstep=degstep,
        ignore_base=ignore_base,
        minplot=minplot,
        maxplot=maxplot,
        offset=offset[i]
    )


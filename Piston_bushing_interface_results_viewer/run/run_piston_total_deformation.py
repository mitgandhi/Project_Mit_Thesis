import piston_total_deformation

filepaths = [
    r'`Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP_FD\Run\Run1\T01_4400n_380dp_100d'
]

labels = ['4400 rpm', '4400 rpm']
m = [80, 80]
lF = [39.5, 39.5]
n = [4400, 4400]
deltap = [380, 380]
offset = [0, 0]

plots = 360
degstep = 1
ognore = 0
minplot = 0
maxplot = 0

for i in range(len(filepaths)):
    print(f"\n=== Piston Total Deformation Case {i+1}: {labels[i]} ===")
    piston_total_deformation(
        filepath=filepaths[i],
        m=m[i],
        lF=lF[i],
        n=n[i],
        deltap=deltap[i],
        plots=plots,
        degstep=degstep,
        ognore=ognore,
        minplot=minplot,
        maxplot=maxplot,
        offset=offset[i]
    )

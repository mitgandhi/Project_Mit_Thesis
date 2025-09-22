# Slipper Gap Force Calculation Documentation

## Overview

The SlipperCalcHolder function calculates forces acting on a slipper gap using different hold-down mechanisms.

Base total force calculation:
```
Total Force (Ftot) = Slipper Force (Fslipper) / Number of pistons (npistons)
```

## Hold-Down Mechanisms

### 1. Passive Hold-Down (SlipHD = 0)

Forces are calculated based on gap height relative to maximum allowable gap.

For each point i (where i = 1,2,3):
```
If xg[i] > hmaxG:
    F[i] = (xg[i] - hmaxG) * HoldDownStiffness
Else:
    F[i] = 0
```

Where:
* xg[i] = Gap height at point i
* hmaxG = Maximum allowable gap height
* HoldDownStiffness = Hold-down stiffness coefficient

### 2. Active Hold-Down (SlipHD = 1)

Force is distributed equally:
```
For each point i (i = 1,2,3):
    F[i] = Ftot / 3
```

### 3. Combined Active and Passive (SlipHD = 2)

Forces depend on maximum gap distance:
```
hT = h + ehd - hgroove * 5e-6
maxdist = maximum(hT) - minimum(hT)
```

#### When maxdist < lower threshold (5μm):
```
For each point i (i = 1,2,3):
    F[i] = Ftot / 3
```

#### When maxdist ≥ lower threshold:

1. Force Components:
```
Fu = ((upper - maxdist) / (upper - lower)) * Ftot
Fp = Ftot - Fu
```

2. Total Force Calculation:
```
For each point i:
    F[i] = (Fu / 3) + x[i] + Additional Hold Down Force

Where x[i] is solved from matrix equation Ax = b:
b = [Fp, Fp*dy, -Fp*dx]

Additional Hold Down Force:
If xg[i] > hmaxG:
    = (xg[i] - hmaxG) * HoldDownStiffness
Else:
    = 0
```

### 4. Legacy Mode (SlipHD = 3)

#### Spring Hold-Down (hmaxG < -999μm)

Forces based on gap distance:
```
If maxdist < lower:
    Fu = Ftot
    Fp = 0
Else If maxdist > upper:
    Fp = Ftot
    Fu = 0
Else:
    Fp = (maxdist - lower) / (upper - lower) * Ftot
    Fu = Ftot - Fp

For each point i:
    F[i] = (Fu / 3) + x[i]

Where x[i] is solved from Ax = b:
b = [Fp, Fp*dy, -Fp*dx]
```

#### Fixed Hold-Down (hmaxG ≥ -999μm)

Forces calculated using contact pressure:
```
contact = maximum(0, hT - hmaxG)
contactp = contact * (Fslipper / N)

Fz = sum(contactp)
Mx = sum(contactp * Ly)
My = sum(-contactp * Lx)

Solve Ax = b where b = [Fz, Mx, My]
F[i] = x[i]
```

## Key Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| Fslipper | Total slipper force | N |
| npistons | Number of pistons | - |
| hmaxG | Maximum allowable gap height | m |
| HoldDownStiffness | Hold-down stiffness coefficient | N/m |
| upper | Upper threshold | 10 μm |
| lower | Lower threshold | 5 μm |
| hgroove | Groove height | m |
| ehd | EHD height adjustment | m |
| Lx, Ly | Position coordinates | m |

## Units and Conventions

* All distances are in meters (m) unless specified in micrometers (μm)
* Forces are measured in Newtons (N)
* Stiffness is measured in Newtons per meter (N/m)
* Variable names match the original code for clarity
* Matrix equations use standard notation: Ax = b
* Arrays are zero-indexed

## Notes

* The function handles four different hold-down mechanisms
* Transitions between mechanisms are discrete based on SlipHD parameter
* Matrix A is assumed to be non-singular for all solutions
* Contact calculations use element-wise operations on arrays

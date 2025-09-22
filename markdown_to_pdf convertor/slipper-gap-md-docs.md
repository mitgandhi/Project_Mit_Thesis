# Slipper Gap Force Calculation Documentation

## Overview

The `SlipperCalcHolder` function calculates forces acting on a slipper gap using different hold-down mechanisms. The base calculation distributes the total force across pistons:

```
F_tot = F_slipper / n_pistons
```

## Hold-Down Mechanisms

### 1. Passive Hold-Down (SlipHD = 0)

Forces are calculated based on gap height relative to maximum allowable gap.

For each point i (where i = 1,2,3):
```
F[i] = k_HD * (x_g[i] - h_maxG)  if x_g[i] > h_maxG
F[i] = 0                         otherwise
```

where:
- `x_g[i]` = Gap height at point i
- `h_maxG` = Maximum allowable gap height
- `k_HD` = Hold-down stiffness coefficient

### 2. Active Hold-Down (SlipHD = 1)

Force is distributed equally:
```
F[i] = F_tot / 3  for i = 1,2,3
```

### 3. Combined Active and Passive (SlipHD = 2)

Forces depend on maximum gap distance:
```
h_T = h + e_hd - h_groove * 5e-6
maxdist = max(h_T) - min(h_T)
```

#### When maxdist < h_lower (5μm)
```
F[i] = F_tot / 3  for i = 1,2,3
```

#### When maxdist ≥ h_lower
Forces combine uniform and point components:

1. Uniform component:
```
F_u = ((h_upper - maxdist) / (h_upper - h_lower)) * F_tot
F_p = F_tot - F_u
```

2. Total force calculation:
```
F[i] = (F_u / 3) + x[i] + F_HD(i)
```

where:
```
System Ax = b solves for x[i]
b = [F_p, F_p*dy, -F_p*dx]

F_HD(i) = k_HD * (x_g[i] - h_maxG)  if x_g[i] > h_maxG
F_HD(i) = 0                         otherwise
```

### 4. Legacy Mode (SlipHD = 3)

#### Spring Hold-Down (h_maxG < -999μm)
Forces based on gap distance:
```
if maxdist < h_lower:
    F_u = F_tot
    F_p = 0
else if maxdist > h_upper:
    F_p = F_tot
    F_u = 0
else:
    F_p = ((maxdist - h_lower) / (h_upper - h_lower)) * F_tot
    F_u = F_tot - F_p

F[i] = (F_u / 3) + x[i]
```

where x[i] is solved from:
```
Ax = b
b = [F_p, F_p*dy, -F_p*dx]
```

#### Fixed Hold-Down (h_maxG ≥ -999μm)
Forces calculated using contact pressure:
```
contact = max(0, h_T - h_maxG)
contact_p = contact * (F_slipper / N)

F_z = sum(contact_p)
M_x = sum(contact_p * L_y)
M_y = sum(-contact_p * L_x)

Solve Ax = b where b = [F_z, M_x, M_y]
F[i] = x[i]
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `F_slipper` | Total slipper force |
| `n_pistons` | Number of pistons |
| `h_maxG` | Maximum allowable gap height |
| `k_HD` | Hold-down stiffness coefficient |
| `h_upper` | Upper threshold (10μm) |
| `h_lower` | Lower threshold (5μm) |
| `h_groove` | Groove height |
| `e_hd` | EHD height adjustment |
| `L_x, L_y` | Position coordinates |

## Notes

- All dimensions are in meters unless specified otherwise
- Forces are in Newtons
- Stiffness coefficient is in N/m

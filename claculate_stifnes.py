# Given values
A1 = 2*8392.8  # mm² (converted to m² later)
A2 = 13314  # mm² (converted to m² later)
E = 2 * 10**5  # MPa = 2 * 10^11 Pa
L = 20  # mm (converted to meters)

# Convert areas to square meters
A1_m2 = A1 * 1e-6  # mm² to m²
A2_m2 = A2 * 1e-6  # mm² to m²

# Calculate individual stiffness values
k1 = (E * A1_m2) / (L * 1e-3)  # Convert L from mm to m
k2 = (E * A2_m2) / (L * 1e-3)

# Calculate total stiffness in series
k_total_series = (k1 * k2) / (k1 + k2)

# Calculate total stiffness in parallel
k_total_parallel = k1 + k2

# Display results
print(k_total_parallel)
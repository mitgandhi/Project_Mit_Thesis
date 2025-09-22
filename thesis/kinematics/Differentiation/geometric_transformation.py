from sympy import Matrix, symbols, cos, sin, tan, simplify, solve, collect


def calculate_normals_and_transformation():
    # Define all symbolic variables
    phi, gamma, alpha, beta = symbols('phi gamma alpha beta')
    L, L2, R = symbols('L1 L2 R')

    # Define normal vectors
    n1 = Matrix([0, 1, -tan(alpha)])
    n0 = Matrix([1, 0, -tan(beta)/cos(alpha)])

    # Calculate cross product n0 × n1
    cross_product = Matrix([
        n0[1] * n1[2] - n0[2] * n1[1],  # i component
        n0[2] * n1[0] - n0[0] * n1[2],  # j component
        n0[0] * n1[1] - n0[1] * n1[0]  # k component
    ])

    # Get transformation matrices
    T01, T12, T23, T34, T45 = create_transformation_matrices()

    # Calculate final transformation
    T_final =  T01*T12 * T23 * T34* T45

    # Extract translation components (x, y, z)
    x = T_final[0, 3]  # Element (1,4) in the matrix
    y = T_final[1, 3]  # Element (2,4) in the matrix
    z = T_final[2, 3]  # Element (3,4) in the matrix

    # Form the equation by substituting x, y, z into the cross product equation
    equation = (cross_product[0] * x +
                cross_product[1] * y +
                cross_product[2] * z)

    print(T_final)
    # Print results
    print("Normal vectors:")
    print("n0 =")
    print(n0)
    print("\nn1 =")
    print(n1)

    print("\nCross product n0 × n1 =")
    print(cross_product)

    print("\nTranslation components from transformation matrix:")
    print("x =")
    print(x)
    print("\ny =")
    print(y)
    print("\nz =")
    print(z)

    print("\nFinal equation:")
    print(simplify(equation))

    return cross_product, equation


def create_transformation_matrices():
    # Define symbolic variables
    phi, gamma = symbols('phi gamma')
    L, L2, R = symbols('L1 L2 R')

    # T01: First transformation matrix
    T01 = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, L],
        [0, 0, 0, 1]
    ])

    # T12: Rotation around Z-axis by phi
    T12 = Matrix([
        [cos(phi), -sin(phi), 0, 0],
        [sin(phi), cos(phi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # T23: Translation in X by R
    T23 = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, R],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # T34: Rotation around Y-axis by gamma
    T34 = Matrix([
        [1, 0, 0, 0],
        [0, cos(gamma), -sin(gamma), 0],
        [0, sin(gamma), cos(gamma), 0],
        [0, 0, 0, 1]
    ])

    # T45: Translation in Z by -L2
    T45 = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -L2],
        [0, 0, 0, 1]
    ])

    return T01, T12, T23, T34, T45


# Calculate and display results
cross_product, final_equation = calculate_normals_and_transformation()


L, L2, R = symbols('L1 L2 R')
phi, gamma, alpha, beta = symbols('phi gamma alpha beta')

L2_solved = solve(final_equation, L2)[0]

# # Simplify the expression
L2_simplified = simplify(L2_solved)
#
print("L2 = ")
print(L2_simplified)
equation1 = simplify( L2_simplified - (L/cos(gamma)))
#
#
collected = collect(equation1, [sin(gamma), cos(gamma)])
print("L3")
print(collected)
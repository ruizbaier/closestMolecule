import numpy as np
from fenics import *
from advection_diffusion_mixed import solve_problem_instance
from mesh_gen import construct_mesh
from mesh_gen import boundary_func
import matplotlib.pyplot as plt

def evaluate_normals(mesh, bdry):
    ds = Measure('ds', domain=mesh, subdomain_data=bdry)
    boundary_mesh = BoundaryMesh(mesh, "exterior", True)
    D=VectorFunctionSpace(boundary_mesh, "CG", 2)


    # Approximate facet normal in a suitable space using projection
    n = FacetNormal(mesh)
    V = VectorFunctionSpace(mesh, "CG", 2)
    u_ = TrialFunction(V)
    v_ = TestFunction(V)
    a = inner(u_, v_) * ds
    l = inner(n, v_) * ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)

    solve(A, nh.vector(), L)
    return nh


class normal_u(UserExpression):
    def __init__(self, nh, **kwargs):
        super().__init__(**kwargs)
        self.nh = nh

    def eval(self, values, x):
        n_eval = self.nh(x)
        values[0] = n_eval[0]
        values[1] = n_eval[1]

    def value_shape(self):
        return (2,)


def approx_flux(D1, c, sigma, gamma, r2, corrections):
    return np.square(4*np.pi*r2)*D1*c*np.exp(-4*np.pi*c*np.power(r2,3)/3)*(boundary_func(sigma, gamma, r2, corrections))



if __name__ == '__main__':
    # The name of the output file for the flux results. Not the numerical solution, see solve_problem_instance() for
    # that output file.
    output_filename = 'exp_test.npy'
    # The mesh to solve the problem for.
    mesh_filename = "exp_boundary_sigma0.1_gamma1"
    # The number of mesh refinements to make.
    number_corrections = 20
    # ******* Model constants ****** #
    # sigma and gamma values must match the boundaries of the mesh in 'mesh_filenames'.
    sigma = 0.1
    gamma = 1
    mesh_folder = "meshes/"
    # Dimensions of the mesh
    r1_max = 5
    r2_min = 0
    r2_max = 5
    r1_min = sigma
    # Volume in r1 direction.
    V1 = 4 * np.pi * np.power(r1_max - sigma, 3) / 3
    # Volume r2 direction (space is actually 6D not 2D).
    V2 = 4 * np.pi * np.power(r2_max, 3) / 3
    # Number of points along the bottom of the mesh boundary
    num_bottom_points = 1000
    r2_values = np.linspace(0, r2_max, num_bottom_points)
    # Number of C molecules.
    Nc = V2
    # Concentration of C molecules.
    c = 10
    # Diffusion coefficients.
    D_BASE = 1
    D1 = 2 * D_BASE
    D2 = 1.5 * D_BASE
    # ********** Time constants ********* #
    dt = 0.1
    t_final = 0.1
    # Toogle to solve steady state or time dependent problem
    steady_state = True
    # Initially no corrections to make to the mesh
    corrections = None
    error = 1
    while error > 0.001:
        results = []
        # Create the new mesh.
        construct_mesh(sigma, gamma, r1_max, r2_max, num_bottom_points, mesh_filename, corrections)
        # Read in mesh.
        mesh = Mesh(mesh_folder + mesh_filename + ".xml")
        bdry = MeshFunction("size_t", mesh, mesh_folder + mesh_filename + "_facet_region.xml")
        nh = evaluate_normals(mesh, bdry)
        # Solve problem and get the flux dotted with the normal of the boundary.
        flux, total_flux = solve_problem_instance(c, t_final, dt, mesh, bdry, sigma, gamma, results, V1, D1, D2, steady_state)
        error = total_flux - 4*np.pi*D1*(c/(c+gamma))*0.1
        print(f'actual error {error}')
        #sigma = (1 - error/(4*np.pi*D1*(c/(c+gamma))))*sigma
        # Now determine the discrepancies between actual flux and approximation.
        r1_values = boundary_func(sigma, gamma, r2_values, corrections)
        approx_fluxes = approx_flux(D1, c, sigma, gamma, r2_values, None)
        actual_fluxes = []
        
        # Iterate through the boundary points
        for i in range(len(r2_values)):
            point = Point(r2_values[i], r1_values[i])
            actual_fluxes.append(V1*4*np.pi*np.square(r1_values[i])*np.dot(flux(point), nh(point)))
        # Compute errors
        #print(actual_fluxes)
        actual_fluxes = np.array(actual_fluxes)


        errors = approx_fluxes - actual_fluxes
        # Compute expected adjustment to height of boundary
        prob_density_contribution = np.square(4*np.pi*r2_values)*D1*c*np.exp(-4*np.pi*c*np.power(r2_values,3)/3)
        # Density might be 0 at some points, so no adjustment needed at this point since error should also be 0
        prob_density_contribution[prob_density_contribution == 0] = 1
        corrections = np.zeros(len(r2_values))
        corrections = errors/prob_density_contribution
        corrections[np.abs(corrections) > np.square(sigma)] = 0
        corrections[np.where(actual_fluxes < 0)] = 0
        boundary = boundary_func(sigma, gamma, r2_values, None)
        correction_stack = np.column_stack((actual_fluxes,approx_fluxes,errors, prob_density_contribution, corrections, boundary))
        for ele in correction_stack:
            print(f'flux {ele[0]:.3f} approx {ele[1]:.3f} error {ele[2]:.3f} density {ele[3]:.3f} correction '
                  f'{ele[4]:.3f} boundary {ele[5]:.3f}')
        input()
        plt.plot(r2_values, r1_values)
        plt.show()



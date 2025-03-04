import numpy as np
from fenics import *
from advection_diffusion_mixed import solve_problem_instance
from mesh_gen import construct_mesh
from scipy.optimize import least_squares
import time

def required_rate(conc, sigma, gamma):
    return 4*np.pi*2*sigma*(conc/(conc + gamma))


def jacobian(parameters, sigma_orig, gamma_orig, c_vals, r1_max, r2_max, num_bottom_points, mesh_filename,
                          t_final, dt, D1, D2, steady_state):
    '''
    First order approx to the function is sigma*(c/(c+gamma)) so derivative with respect to sigma is simply: c/(c+gamma)
    while derivative with respect to gamma is: -sigma*c/(c+gamma)^2

    '''
    sigma = parameters[0]
    gamma = parameters[1]
    return np.transpose(np.array([c_vals/(c_vals + gamma), -sigma*c_vals/np.square(c_vals + gamma)]))




def finite_element_fluxes(parameters, sigma_orig, gamma_orig, c_vals, r1_max, r2_max, num_bottom_points, mesh_filename,
                          t_final, dt, D1, D2, steady_state):
    sigma = parameters[0]
    gamma = parameters[1]
    print(f'New sigma: {sigma} new gamma: {gamma}')
    # Volume in r1 direction.
    V1 = 4 * np.pi * np.power(r1_max - sigma, 3) / 3
    # Create the new mesh.
    construct_mesh(sigma, gamma, r1_max, r2_max, num_bottom_points, mesh_filename)
    print('**************************************************************************')
    # Read in mesh.
    mesh = Mesh(mesh_folder + mesh_filename + ".xml")
    bdry = MeshFunction("size_t", mesh, mesh_folder + mesh_filename + "_facet_region.xml")
    # Accumulate the total flux for a series of concentrations for the current mesh.
    finite_ele_fluxes = []
    for c in c_vals:
        print(f'Current concentration {c}')
        # Solve problem and get the flux dotted with the normal of the boundary.
        flux, total_flux = solve_problem_instance(c, t_final, dt, mesh, bdry, sigma, gamma, [], V1, D1, D2,
                                                  steady_state)
        finite_ele_fluxes.append(total_flux)
    finite_ele_fluxes = np.array(finite_ele_fluxes)
    # Now compute the residuals to the desired reaction rate.
    print(required_rate(c_vals, sigma_orig, gamma_orig) - finite_ele_fluxes)
    return required_rate(c_vals, sigma_orig, gamma_orig) - finite_ele_fluxes


if __name__ == '__main__':
    # The name of the output file for the flux results. Not the numerical solution, see solve_problem_instance() for
    # that output file.
    output_filename = 'exp_test.npy'
    # The mesh to solve the problem for.
    mesh_filename = "exp_boundary_correction"
    # ******* Model constants ****** #
    mesh_folder = "meshes/"
    sigma_orig = 0.1
    gamma_orig = 1
    # Dimensions of the mesh
    r1_max = 5
    r2_max = 5
    # Number of points along the bottom of the mesh boundary
    num_bottom_points = 1000
    # Diffusion coefficients.
    D_BASE = 1
    D1 = 2 * D_BASE
    D2 = 1.5 * D_BASE
    # ********** Time constants ********* #
    dt = 0.1
    t_final = 0.1
    # Toogle to solve steady state or time dependent problem
    steady_state = True
    # Allow looping through multiple concentrations
    # Concentration of C molecules.
    c_vals = np.arange(1, 5, 1)
    sigma_init = 0.1
    gamma_init = 1.0
    sigma_upper = 0.15
    gamma_upper = 1.5
    init_guess = np.array([sigma_init, gamma_init])
    start_time = time.time()
    sol = least_squares(fun=finite_element_fluxes, x0=init_guess, jac=jacobian,
                        bounds=([0, 0], [sigma_upper, gamma_upper]), args=[sigma_orig, gamma_orig, c_vals,
                                                                                      r1_max, r2_max, num_bottom_points,
                                                                                      mesh_filename, t_final, dt, D1,
                                                                                      D2, steady_state], method='lm')
    print('******************************************************************************')
    print(f'Elapsed time: {time.time() - start_time}')
    print(f'Solution parameters: {sol.x} cost: {sol.cost} residuals: {sol.fun}')




import numpy as np
from fenics import *
from advection_diffusion_mixed import solve_problem_instance
from mesh_gen import construct_mesh
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time

def required_rate(conc, sigma, gamma):
    return 4*np.pi*2*sigma*(conc/(conc + gamma))

if __name__ == '__main__':
    # The name of the output file for the flux results. Not the numerical solution, see solve_problem_instance() for
    # that output file.
    output_filename = 'exp_test.npy'
    # The mesh to solve the problem for.
    mesh_filename = "exp_boundary_sigma0.1_gamma1"
    # The number of mesh refinements to make.
    number_corrections = 20
    # ******* Model constants ****** #
    mesh_folder = "meshes/"
    sigma_orig = 0.1
    gamma_orig = 1
    # Dimensions of the mesh
    r1_max = 5
    r2_min = 0
    r2_max = 5
    r1_min = sigma_orig
    # Volume in r1 direction.
    V1 = 4 * np.pi * np.power(r1_max - r1_min, 3) / 3
    # Volume r2 direction (space is actually 6D not 2D).
    V2 = 4 * np.pi * np.power(r2_max, 3) / 3
    # Number of points along the bottom of the mesh boundary
    num_bottom_points = 1000
    r2_values = np.linspace(0, r2_max, num_bottom_points)
    # Number of C molecules.
    Nc = V2
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
    error_tolerance = 0.001
    learning_rate = 10
    corrected_parameters = []
    c_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sigma = sigma_orig
    gamma = gamma_orig
    sigma_error = 1
    gamma_error = 1
    start_time = time.time()
    while np.abs(sigma_error) > error_tolerance or np.abs(gamma_error) > error_tolerance:
        results = []
        # Create the new mesh.
        construct_mesh(sigma, gamma, r1_max, r2_max, num_bottom_points, mesh_filename)
        # Read in mesh.
        mesh = Mesh(mesh_folder + mesh_filename + ".xml")
        bdry = MeshFunction("size_t", mesh, mesh_folder + mesh_filename + "_facet_region.xml")
        # Accumulate the total flux for a series of concentrations for the current mesh.
        finite_ele_fluxes = []
        for c in c_vals:
            print(f'Current concentration {c}')
            # Solve problem and get the flux dotted with the normal of the boundary.
            flux, total_flux = solve_problem_instance(c, t_final, dt, mesh, bdry, sigma, gamma, results, V1, D1, D2,
                                                      steady_state)
            finite_ele_fluxes.append(total_flux)
        # Now fit expected reaction function to the finite element fluxes to get the effective sigma and gamma values
        finite_ele_fluxes = np.array(finite_ele_fluxes)
        popt, pcov = curve_fit(required_rate, c_vals, finite_ele_fluxes)
        finite_element_sigma = popt[0]
        finite_element_gamma = popt[1]
        # Compute error in both sigma and gamma based on the effective values obtained
        sigma_error = sigma_orig - finite_element_sigma
        gamma_error = gamma_orig - finite_element_gamma
        print(f'Finite element sigma: {finite_element_sigma} sigma error: {sigma_error} finite element gamma: '
              f'{finite_element_gamma} gamma error: {gamma_error}')
        if np.abs(sigma_error) > error_tolerance:
            # Only update parameter if still outside tolerance
            sigma = sigma + sigma_error
        if np.abs(gamma_error) > error_tolerance:
            # Only update parameter if still outside tolerance
            gamma = gamma + gamma_error
        print(f'New sigma: {sigma} new gamma: {gamma}')
        print('**************************************************************************')


    print('******************************************************************************')
    print(f'Elapsed time: {time.time() - start_time}')
    print(f'Original sigma: {sigma_orig} adjusted: {sigma} finite element: {finite_element_sigma} error: {sigma_error}')
    print(f'Original gamma: {gamma_orig} adjusted: {gamma} finite element: {finite_element_gamma} error: {gamma_error}')



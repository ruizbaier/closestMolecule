import numpy as np
from fenics import *
from advection_diffusion_mixed import solve_problem_instance
from scipy.optimize import least_squares
import subprocess
import gmsh
import os
import time


def required_rate(conc, sigma, gamma):
    return 4*np.pi*2*sigma*(conc/(conc + gamma))


def jacobian(parameters, sigma_orig, gamma_orig, c_vals, r1_max, mesh_folder, mesh_filename,
                          t_final, dt, D1, D2, steady_state):
    '''
    First order approx to the function is sigma*(c/(c+gamma)) so derivative with respect to sigma is simply: c/(c+gamma)
    while derivative with respect to gamma is: -sigma*c/(c+gamma)^2

    '''
    sigma = parameters[0]
    gamma = parameters[1]
    return np.transpose(np.array([c_vals/(c_vals + gamma), -sigma*c_vals/np.square(c_vals + gamma)]))


def create_mesh(sigma, gamma, mesh_folder, mesh_filename):
    print(f'Constructing mesh for {sigma=} {gamma=}')
    # Creates the .mesh file for the new parameters
    edp_filename = os.path.join(mesh_folder, "automated_exp_meshing.edp")
    result = subprocess.run(["FreeFem++", edp_filename, str(sigma), str(gamma)], capture_output=True, text=True)
    #print("stdout edp:\n", result.stdout)
    #print("stderr edp:\n", result.stderr)
    # Convert .mesh to .msh
    gmsh.initialize()
    gmsh.merge(mesh_filename)  # or gmsh.open

    for i in range(1, 4):
        E = gmsh.model.getEntities(i)
        for ei in E:
            gmsh.model.addPhysicalGroup(i, [ei[1]], ei[1])

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    msh_filename = os.path.join(mesh_filename[:-4] + "msh")
    gmsh.write(msh_filename)
    gmsh.finalize()
    # Convert .msh to .xml
    result = subprocess.run(
        ["dolfin-convert", msh_filename, msh_filename[:-3]+'xml'],
        capture_output=True,
        text=True
    )
    #print("stdout dolfin-convert:\n", result.stdout)
    #print("stderr dolfin-convert:\n", result.stderr)
    print(f'Mesh constructed for {sigma=} {gamma=}')
    print('########################################')


def finite_element_fluxes(parameters, sigma_orig, gamma_orig, c_vals, r1_max, mesh_folder, mesh_filename,
                          t_final, dt, D1, D2, steady_state):
    sigma = parameters[0]
    gamma = parameters[1]
    print(f'New sigma: {sigma} new gamma: {gamma}')
    # Volume in r1 direction.
    V1 = 4 * np.pi * np.power(r1_max - sigma, 3) / 3
    # Create the new mesh.
    create_mesh(sigma, gamma, mesh_folder, mesh_filename+'.mesh')
    print('**************************************************************************')
    # Read in mesh (filename comes without file suffix i.e. no .mesh etc.).
    mesh = Mesh(mesh_filename + ".xml")
    bdry = MeshFunction("size_t", mesh, mesh_filename + "_facet_region.xml")
    # Accumulate the total flux for a series of concentrations for the current mesh.
    finite_ele_fluxes = []
    # The location the solution to each instance of the problem is written to.
    output_folder = 'corrections/outputs/'
    for c in c_vals:
        print(f'Current concentration {c}')
        # Solve problem and get the flux dotted with the normal of the boundary.
        finite_element_filename = os.path.join(output_folder, f'exp_boundary_c{c}_sigma{sigma:.2f}_gamma{gamma:.2f}.xdmf')
        flux, total_flux = solve_problem_instance(c, t_final, dt, mesh, bdry, sigma, gamma, [], V1, D1, D2, finite_element_filename, steady_state)
        finite_ele_fluxes.append(total_flux)
    finite_ele_fluxes = np.array(finite_ele_fluxes)
    # Now compute the residuals to the desired reaction rate.
    print(f'Finite fluxes: {finite_ele_fluxes} required rates: {required_rate(c_vals, sigma_orig, gamma_orig)} '
          f'residuals: {finite_ele_fluxes - required_rate(c_vals, sigma_orig, gamma_orig)}')
    return finite_ele_fluxes - required_rate(c_vals, sigma_orig, gamma_orig)


if __name__ == '__main__':
    # The mesh to solve the problem for is generated automatically (see create_mesh above).
    mesh_folder = 'meshes/'
    mesh_filename = os.path.join(mesh_folder, 'exp_boundary_auto')
    # Mesh parameters.
    sigma_orig = 0.1
    gamma_orig = 1
    # Dimensions of the mesh.
    r1_max = 5
    r2_max = 5
    # Diffusion coefficients.
    D_A = 1
    D_B = 1
    D_C = 5
    D1 = D_A + D_B
    D2 = D_C + 1/(1/D_A + 1/D_B)
    # Time constants.
    dt = 0.1
    t_final = 0.1
    # Toogle to solve steady-state or time-dependent problem.
    steady_state = True
    # Concentration of C molecules.
    c_vals = np.array([1,2,4,8])
    sigma_init = 0.09867282
    gamma_init = 1.20166547
    init_guess = np.array([sigma_init, gamma_init])
    # Exactly 0 can cause some issues.
    bounds = ([1e-10, 1e-10], [np.inf, np.inf])
    start_time = time.time()
    # Solves the optimisation problem.
    sol = least_squares(fun=finite_element_fluxes,
                        args=[sigma_orig, gamma_orig, c_vals, r1_max, mesh_folder, mesh_filename, t_final, dt, D1, D2, steady_state],
                        x0=init_guess, jac=jacobian, method='trf', bounds=bounds, max_nfev=25,
                        x_scale=[sigma_orig / 10, gamma_orig / 10])
    print('******************************************************************************')
    print(f'Elapsed time: {time.time() - start_time}')
    print(f'Solution parameters: {sol.x} cost: {sol.cost} residuals: {sol.fun}')




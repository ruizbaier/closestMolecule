'''
 - r2 ->

______  ^
|     | |
|     | r1
|_____| |

'''
import numpy as np
from fenics import *
import matplotlib.pyplot as plt

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

np.seterr(all='raise')
np.seterr(under='ignore')

def solve_problem_instance(concentration, t_final, dt, mesh, bdry, sigma, gamma, results, steady_state = False):
    """
    Solves a specific instance of the problem. Writes the numerical solution to an xdmf file
    and returns some flux calculations in the 'results' array.

    Parameters
    ----------
    concentration: float
        Represents the concentration of C molecules within the system.
    t_final: float
        Defines the period [0, t_final] over which the problem is to be solved.
    dt: float
        The time step for the numerical solver.
    mesh: Mesh
        The mesh over which the finite element is required.
    bdry: MeshFucntion
        The mesh function that marks the boudnaries of 'mesh'.
    sigma: float
        The maximum height of the bottom/reaction boundary in the mesh. We assume here the boundary is
        ~ sigma*exp(-4*pi*gamma*r2^3/3).
    gamma: float
        The rate of decay for the bottom boundary in the mesh.
    results: list[float]
        Initially empty this list is populated with the flux computations (namely the net flux over the bottom boundary)
        at the end of the computation.
    steady_state: bool
        Whether to solve the steady-state problem or not.
    """
    r2, r1 = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    weight = (4 * pi * r1 * r2) ** 2
    r2_vec = Constant((1, 0))
    r1_vec = Constant((0, 1))
    r1_matrix = as_tensor([[0, 0], [0, 1]])
    r2_matrix = as_tensor([[1, 0], [0, 0]])

    # mesh labels
    right = 21
    top = 22
    left = 23
    bottom = 24

    # ******** Finite dimensional spaces ******** #
    deg = 3
    P0 = FiniteElement('CG', mesh.ufl_cell(), deg)
    P1 = FiniteElement('CG', mesh.ufl_cell(), deg)
    mixed_space = FunctionSpace(mesh, MixedElement([P0, P1]))
    # ******** Initialise output file ******** #
    output_file = XDMFFile(f'outputs/exp_solution_c{concentration}_sigma{sigma}_gamma{gamma}.xdmf')
    output_file.parameters["rewrite_function_mesh"] = False
    output_file.parameters["flush_output"] = True
    output_file.parameters["functions_share_mesh"] = True
    # ******** Initialise time evolution ******** #
    if steady_state:
        t_final = 0
    t = 0
    # ********* test and trial functions ******** #
    v1, v2 = TestFunctions(mixed_space)
    u = Function(mixed_space)
    du = TrialFunction(mixed_space)
    p, q = split(u)

    # ********* initial and boundary conditions ******** #
    p_initial = PInitial(concentration, sigma, gamma, degree=deg)
    q_initial = QInitial(concentration, degree=deg)

    p_old = interpolate(p_initial, mixed_space.sub(0).collapse())

    # the formulation for p is primal, so the Dirichlet conditions remain so.
    p_right = p_initial
    p_top = p_initial
    p_bottom = Constant(0.)

    # Boundary condition on the left is a no flux condition which is accounted for in the weak formulation.
    p_right_boundary_condition = DirichletBC(mixed_space.sub(0), p_right, bdry, right)
    p_top_boundary_condition = DirichletBC(mixed_space.sub(0), p_top, bdry, top)
    p_bottom_boundary_condition = DirichletBC(mixed_space.sub(0), p_bottom, bdry, bottom)

    # the formulation for q is primal, so the Dirichlet conditions remain so
    q_right = q_initial

    q_right_boundary_condition = DirichletBC(mixed_space.sub(1), q_right, bdry, right)

    bc = [p_right_boundary_condition, p_top_boundary_condition, p_bottom_boundary_condition, q_right_boundary_condition]

    # (approximate) steady-state solution used for comparison
    p_steady_state = PInitial(concentration, sigma, gamma, degree=deg)
    p_flux_approx = FluxApprox(concentration,  sigma, gamma, D1, degree=deg)

    test_space = FunctionSpace(mesh, "CG", deg)
    p_approx = 4*np.pi*r2**2*interpolate(p_steady_state, test_space)
    flux_approx = interpolate(p_flux_approx, test_space)


    # ********* Weak forms ********* #
    if not steady_state:
        # Time dependent weak form.
        FF = (p - p_old) / dt * v1 * weight * dx \
             + dot(dMatrix * (grad(p) + Gstar(p, q, concentration, r2) * r2_vec), grad(v1)) * weight * dx \
             + Dx(q, 0) * v2 * weight * dx + r2 ** 2 * p * v2 * weight * dx
    else:
        # Steady-state weak form.
        FF = dot(dMatrix * (grad(p) + Gstar(p, q, concentration, r2) * r2_vec), grad(v1)) * weight * dx \
             + Dx(q, 0) * v2 * weight * dx + r2 ** 2 * p * v2 * weight * dx

    # Initialise solver
    Tang = derivative(FF, u, du)
    problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs=bc)
    solver = NonlinearVariationalSolver(problem)
    solver_type = 'newton'
    solver.parameters['nonlinear_solver'] = solver_type
    solver.parameters[solver_type + '_solver']['linear_solver'] = 'mumps'
    solver.parameters[solver_type + '_solver']['absolute_tolerance'] = 1e-10
    solver.parameters[solver_type + '_solver']['relative_tolerance'] = 1e-10
    solver.parameters[solver_type + '_solver']['maximum_iterations'] = 10

    vector_space = FunctionSpace(mesh, "RT", deg)

    while (t <= t_final):
        print("t=%.3f" % t)
        solver.solve()
        p_h, q_h = u.split()
        adjusted_p_h = project(4*np.pi*r2**2*p_h, test_space)
        # Compute flux
        flux = project(-4*np.pi*r2**2*(dMatrix*(grad(p_h) + r2*r2*p_h/q_h*p_h*r2_vec)), vector_space)
        # Save the actual solution
        adjusted_p_h.rename("p", "p")
        q_h.rename("q", "q")
        flux.rename("flux", "flux")
        output_file.write(adjusted_p_h, t)
        output_file.write(q_h, t)
        output_file.write(flux, t)
        # Compare solution to approximation
        difp = project((adjusted_p_h - p_approx), test_space)
        difp.rename("dif", "dif")
        output_file.write(difp, t)
        # Update the solution for next iteration
        assign(p_old, p_h)
        t += dt

    # Try to compute flux over boundary
    ds = Measure('ds', domain=mesh, subdomain_data=bdry)
    total_flux = V1 * assemble(dot(flux, n) * 4 * pi * r1 ** 2 * ds(bottom))
    total_r1_flux = V1 * assemble(dot(flux, r1_matrix * n) * 4 * pi * r1 ** 2 * ds(bottom))
    total_r2_flux = V1 * assemble(dot(flux, r2_matrix * n) * 4 * pi * r1 ** 2 * ds(bottom))
    total_approx_flux = D1 * V1 * assemble(Dx(p_approx, 1) * 4 * pi * r1 ** 2 * ds(bottom))

    print(f'r1 flux: {total_r1_flux} r2 flux: {total_r2_flux} total flux: {total_flux} approximate flux: '
          f'{total_approx_flux} error: {total_flux - total_approx_flux} sigma: {sigma} sigma sq: {sigma ** 2}')
    results.append([sigma, gamma, concentration, total_r1_flux, total_r2_flux, total_flux, total_approx_flux])
    return p_h, q_h, q_dif


def exp_reaction_boundary(sigma, gamma, r2_val):
    return sigma#*np.exp(-4/3*np.pi*gamma*np.power(r2_val, 3))


def flat_reaction_boundary(r2_val):
    return sigma


def Gstar(p, q, c, r2):
    """
    The correct non-linear term is (r2*p)^2/q but for sufficiently large r2 this limits to 4*np.pi*c*p*r2**2.
    Using this approximation avoids the floating point errors that arise from computing (r2*p)^2/q when p and q are
    very small.
    """
    return conditional(gt(abs(1/(4*np.pi)*p - q), 0.001), (p/q)*p*r2**2, 4*np.pi*c*p*r2**2)


class PInitial(UserExpression):
    """
    Custom expression to avoid the issues Fenics seems to have with exp().
    """
    def __init__(self, conc, sigma, gamma, **kwargs):
        super().__init__(**kwargs)
        self.conc = conc
        self.sigma = sigma
        self.gamma = gamma

    def eval(self, values, x):
        if x[1] > 0:
            values[0] = self.conc/V1*np.exp(-4/3*np.pi*self.conc*np.power(x[0],3))*(1 - exp_reaction_boundary(self.sigma,
                                                                                                          self.gamma,
                                                                                                          x[0])/x[1])
        else:
            values[0] = self.conc / V1 * np.exp(-4 / 3 * np.pi * self.conc * np.power(x[0], 3))

    def value_shape(self):
        return ()


class Flux(UserExpression):
    """
    Custom expression to avoid the issues Fenics seems to have with exp().
    """
    def __init__(self, p, q, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.q = q
        self.r1_vec = Constant((0, 1))
        self.r2_vec = Constant((1, 0))

    def eval(self, values, x):
        gradient = -4*np.pi*x[0]**2*grad(self.p)
        values[0] = D2*(dot(gradient,self.r2_vec) + (x[0]**2*self.p**2/self.q))
        values[1] = D1*dot(gradient,self.r1_vec)

    def value_shape(self):
        return (2,)

class QInitial(UserExpression):
    """
    Custom expression to avoid the issues Fenics seems to have with exp().
    """
    def __init__(self, conc, **kwargs):
        super().__init__(**kwargs)
        self.conc = conc

    def eval(self, values, x):
        values[0] = 1/(4*np.pi*V1)*np.exp(-4/3*np.pi*self.conc*np.power(x[0], 3))

    def value_shape(self):
        return ()



class FluxApprox(UserExpression):
    """
    Custom expression to avoid the issues Fenics seems to have with exp().
    """
    def __init__(self, conc, sigma, gamma, D1, **kwargs):
        super().__init__(**kwargs)
        self.conc = conc
        self.sigma = sigma
        self.gamma = gamma
        self.D1 = D1

    def eval(self, values, x):
        if np.square(x[1]) > 0:
            try:
                values[0] = -self.D1*self.conc / V1 * np.exp(-4 / 3 * np.pi * self.conc * np.power(x[0], 3)) * \
                            (exp_reaction_boundary(self.sigma, self.gamma, x[0]) / np.square(x[1]))
            except:
                print(f'r1: {x[1]}, r2: {x[0]}')
        else:
            values[0] = 0

    def value_shape(self):
        return ()


if __name__ == '__main__':
    # The name of the output file for the flux results. Not the numerical solution, see solve_problem_instance() for
    # that output file.
    output_filename = 'flat_test'
    # The meshes to solve the problem for. Represented as an array to allow scanning through multiple problem instances.
    mesh_filenames = ["flat_boundary_sigma0.1_r1max_4"]

    # ******* Model constants ****** #
    # Sigma and gamma values must match the boundaries of the meshes in 'mesh_filenames'.
    sigmas = np.array([0.1])
    gammas = [1]
    mesh_folder = "meshes/"
    # Dimensions of the mesh
    r1_max = 4
    r2_min = 0
    r2_max = 5
    R = r1_max
    # Volume r2 direction (space is actually 6D not 2D).
    V2 = 4 * np.pi * np.power(r2_max, 3) / 3
    # Number of C molecules.
    Nc = V2
    # Concentration of C molecules.
    c = 10
    # Diffusion coefficients.
    D_BASE = 1
    D1 = 2 * D_BASE
    D2 = 1.5 * D_BASE
    dMatrix = as_tensor([[D2, 0], [0, D1]])
    # ********** Time constants ********* #
    dt = 0.1
    t_final = 1
    # Toogle to solve steady state or time dependent problem
    steady_state = True
    for i in range(len(mesh_filenames)):
        results = []
        mesh_filename = mesh_filenames[i]
        sigma = sigmas[i]
        gamma = gammas[i]
        r1_min = sigma
        # Volume in r1 direction.
        V1 = 4 * np.pi * np.power(R - sigma, 3) / 3
        # Read in mesh.
        mesh = Mesh(mesh_folder + mesh_filename + ".xml")
        bdry = MeshFunction("size_t", mesh, mesh_folder + mesh_filename + "_facet_region.xml")
        # Solve problem
        solve_problem_instance(c, t_final, dt, mesh, bdry, sigma, gamma, results, steady_state)
        print(results)
        results = np.array(results)
        if len(mesh_filenames) > 0:
            np.save(f'{output_filename}{i}.npy', results)
        else:
            # Only single mesh so just write final result
            np.save(f'{output_filename}.npy', results)
    # Compile all the results if there are multiple meshes.
    if len(mesh_filenames) > 0:
        final_results = None
        for i in range(len(sigmas)):
            data = np.load(f'{output_filename}{i}.npy')
            if final_results is None:
                final_results = data
            else:
                final_results = np.concatenate([final_results, data])
        np.save(f'{output_filename}.npy', final_results)
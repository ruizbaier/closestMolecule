'''
Runs a convergence test for the flat boundary version of the problem.
'''

import numpy as np
from fenics import *
from create_flat_boundary import create_mesh
import math



parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4


def flat_reaction_boundary():
    return sigma


def Gstar(p, q, c, r2):
    """
    The correct non-linear term is (r2*p)^2/q but for sufficiently large r2 this limits to 4*np.pi*c*p*r2**2.
    Using this approximation avoids the floating point errors that arise from computing (r2*p)^2/q when p and q are
    very small.
    """
    return conditional(gt(abs((1/(4*np.pi)*p) - q), 0.000001), (p/q)*p*r2**2, 4*np.pi*c*p*r2**2)


class PInitial(UserExpression):
    """
    Custom expression to avoid the issues Fenics seems to have with exp().
    """
    def __init__(self, conc, sigma, right=False, **kwargs):
        super().__init__(**kwargs)
        self.conc = conc
        self.sigma = sigma
        self.right = right

    def eval(self, values, x):
        if x[1] > 0 or not self.right:
            values[0] = self.conc/V1*np.exp(-4/3*np.pi*self.conc*np.power(x[0],3))*(1 - flat_reaction_boundary()/x[1])
        else:
            values[0] = self.conc / V1 * np.exp(-4 / 3 * np.pi * self.conc * np.power(x[0], 3))

    def value_shape(self):
        return ()


class SInitial(UserExpression):
    """
    Custom expression to avoid the issues Fenics seems to have with exp().
    """
    def __init__(self, conc, sigma, **kwargs):
        super().__init__(**kwargs)
        self.conc = conc
        self.sigma = sigma

    def eval(self, values, x):
        values[0] = 0
        if np.square(x[1]) > 0:
            values[1] = -self.conc/V1*np.exp(-4/3*np.pi*self.conc*np.power(x[0],3))*\
                        (flat_reaction_boundary()/np.square(x[1]))
        else:
            values[1] = 0

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
    def __init__(self, conc, sigma, D1, **kwargs):
        super().__init__(**kwargs)
        self.conc = conc
        self.sigma = sigma
        self.D1 = D1

    def eval(self, values, x):
        values[0] = 0
        values[1] = -self.D1*self.conc/V1*np.exp(-4/3*np.pi*self.conc*np.power(x[0], 3))*\
                    (flat_reaction_boundary()/np.square(x[1]))

    def value_shape(self):
        return (2,)


def solve_problem_instance(concentration, t_final, dt, mesh, bdry, sigma, deg, steady_state = False):
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
    deg = deg
    P0 = FiniteElement('CG', mesh.ufl_cell(), deg)
    P1 = FiniteElement('CG', mesh.ufl_cell(), deg)
    mixed_space = FunctionSpace(mesh, MixedElement([P0, P1]))
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
    p_initial = PInitial(concentration, sigma, degree=deg)
    q_initial = QInitial(concentration, degree=deg)

    p_old = interpolate(p_initial, mixed_space.sub(0).collapse())

    # the formulation for p is primal, so the Dirichlet conditions remain so.
    p_right = PInitial(concentration, sigma, right=True, degree=deg)
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
    p_steady_state = PInitial(concentration, sigma, degree=deg)

    test_space = FunctionSpace(mesh, "CG", deg+3)
    p_approx = interpolate(p_steady_state, test_space)
    p_flux_approx = FluxApprox(concentration, sigma, D1, degree=deg-1)
    vector_space = VectorFunctionSpace(mesh, "DG", deg-1)
    flux_approx = interpolate(p_flux_approx, vector_space)


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
    solver.parameters[solver_type + '_solver']['absolute_tolerance'] = 1e-9
    solver.parameters[solver_type + '_solver']['relative_tolerance'] = 1e-9
    solver.parameters[solver_type + '_solver']['maximum_iterations'] = 10


    while (t <= t_final):
        print("t=%.3f" % t)
        solver.solve()
        p_h, q_h = u.split()
        flux = project((dMatrix * (grad(p_h)) + D2*r2 * r2 * p_h / q_h * p_h * r2_vec),
                       vector_space)
        # Update the solution for next iteration
        assign(p_old, p_h)
        t += dt

    return p_h, p_approx, flux, flux_approx



def solve_problem_instance_mixed(concentration, t_final, dt, mesh, bdry, sigma, deg, steady_state = False):
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
    deg = deg
    P0 = FiniteElement('DG', mesh.ufl_cell(), deg - 1)
    RT1 = FiniteElement('RT', mesh.ufl_cell(), deg)
    P1 = FiniteElement('CG', mesh.ufl_cell(), deg)
    mixed_space = FunctionSpace(mesh, MixedElement([P0, RT1, P1]))

    # ******** Initialise time evolution ******** #
    if steady_state:
        t_final = 0
    t = 0
    # ********* test and trial functions ******** #
    v1, tau, v2 = TestFunctions(mixed_space)
    u = Function(mixed_space)
    du = TrialFunction(mixed_space)
    p, s, q = split(u)

    # ********* initial and boundary conditions ******** #
    # ********* initial and boundary conditions ******** #
    p_initial = PInitial(concentration, sigma, degree=deg-1)
    q_initial = QInitial(concentration, degree=deg)

    p_old = interpolate(p_initial, mixed_space.sub(0).collapse())

    # as the formulation for p-fluxp is mixed, the boundary condition for p becomes natural and the boundary condition
    # for the flux becomes essential (dirichlet)
    p_right = PInitial(concentration, sigma, right=True, degree=deg)
    p_top = p_initial
    p_bottom_boundary_condition = DirichletBC(mixed_space.sub(0), Constant(0), bdry, bottom)

    # Boundary conditions for s are complementary to those of p:
    s_left_boundary_condition = DirichletBC(mixed_space.sub(1), (Constant(0), Constant(0)), bdry,
                                            left)

    # the formulation for q is primal, so the Dirichlet conditions remain so
    q_right = q_initial

    q_right_boundary_condition = DirichletBC(mixed_space.sub(2), q_right, bdry, right)

    # here we only list the Dirichlet conditions
    bc = [s_left_boundary_condition, q_right_boundary_condition, p_bottom_boundary_condition]
    ds = Measure('ds', domain=mesh, subdomain_data=bdry)

    # ********* Weak forms ********* #
    if not steady_state:
        # Time dependent weak form.
        FF = (p - p_old) / dt * v1 * weight * dx \
             + ((D1 / r1 ** 2) * Dx(r1 ** 2 * (dot(s, r1_vec)), 1) + (D2 / r2 ** 2) * Dx(r2 ** 2 * (dot(s, r2_vec)),
                                                                                         0)) * v1 * weight * dx \
             + dot(s + (Gstar(p, q, concentration, r2) * r2_vec), tau) * weight * dx \
             - p * ((1 / r1 ** 2) * Dx(r1 ** 2 * (dot(tau, r1_vec)), 1) + (1 / r2 ** 2) * Dx(
            r2 ** 2 * (dot(tau, r2_vec)), 0)) * weight * dx \
             + p_right * dot(tau, n) * weight * ds(right) \
             + p_top * dot(tau, n) * weight * ds(top) \
             + Dx(q, 0) * v2 * weight * dx + r2 ** 2 * p * v2 * weight * dx

        # Last term is from the boundary condition for q which states dq/dr2 = 0 on the inner boundary.
    else:
        # Steady-state weak form.
        FF = ((D1 / r1 ** 2) * Dx(r1 ** 2 * (dot(s, r1_vec)), 1) + (D2 / r2 ** 2) * Dx(r2 ** 2 * (dot(s, r2_vec)),
                                                                                         0)) * v1 * weight * dx \
             + dot(s + (Gstar(p, q, concentration, r2) * r2_vec), tau) * weight * dx \
             - p * ((1 / r1 ** 2) * Dx(r1 ** 2 * (dot(tau, r1_vec)), 1) + (1 / r2 ** 2) * Dx(
            r2 ** 2 * (dot(tau, r2_vec)), 0)) * weight * dx \
             + p_right * dot(tau, n) * weight * ds(right) \
             + p_top * dot(tau, n) * weight * ds(top) \
             + Dx(q, 0) * v2 * weight * dx + r2 ** 2 * p * v2 * weight * dx

    # Initialise solver
    Tang = derivative(FF, u, du)
    problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs=bc)
    solver = NonlinearVariationalSolver(problem)
    solver_type = 'newton'
    solver.parameters['nonlinear_solver'] = solver_type
    solver.parameters[solver_type + '_solver']['linear_solver'] = 'mumps'
    solver.parameters[solver_type + '_solver']['absolute_tolerance'] = 1e-9
    solver.parameters[solver_type + '_solver']['relative_tolerance'] = 1e-9
    solver.parameters[solver_type + '_solver']['maximum_iterations'] = 10

    # (approximate) steady-state solution used for comparison
    p_steady_state = PInitial(concentration, sigma, degree=deg+3)
    p_flux_approx = SInitial(concentration, sigma, degree=deg+3)

    p_approx = interpolate(p_steady_state, mixed_space.sub(0).collapse())
    p_flux_approx = interpolate(p_flux_approx, mixed_space.sub(1).collapse())

    while (t <= t_final):
        print("t=%.3f" % t)
        solver.solve()
        p_h, s_h, q_h = u.split()
        # Compute flux
        assign(p_old, p_h)
        t += dt

    return p_h, p_approx, s_h, p_flux_approx



if __name__ == '__main__':
    deg = 4
    hh = []
    eu = []
    ru = []
    e0 = []
    r0 = []
    ru.append(0)
    r0.append(0)
    # The meshes to solve the problem for. Represented as an array to allow scanning through multiple problem instances.
    mesh_filenames = ["rect_boundary"]
    mesh_resolutions = [10, 20, 30, 40]
    # ******* Model constants ****** #
    # Sigma and gamma values must match the boundaries of the meshes in 'mesh_filenames'.
    sigmas = np.array([0.1])
    mesh_folder = "meshes/"
    # Dimensions of the mesh
    r1_max = 5
    r2_min = 0
    r2_max = 5
    R = r1_max
    # Volume r2 direction (space is actually 6D not 2D).
    V2 = 4 * np.pi * np.power(r2_max, 3) / 3
    # Number of C molecules.
    Nc = V2
    # Concentration of C molecules.
    c = np.array([1])
    # Diffusion coefficients.
    D_BASE = 1
    D1 = 2 * D_BASE
    D2 = 1.5 * D_BASE
    dMatrix = as_tensor([[D2, 0], [0, D1]])
    # ********** Time constants ********* #
    dt = 0.1
    t_final = 0.1
    # Toogle to solve steady state or time dependent problem
    steady_state = True
    for i in range(len(mesh_resolutions)):
        print(f'Current mesh resolution {i}')
        mesh_filename = mesh_filenames[0]
        sigma = sigmas[0]
        r1_min = sigma
        # Create mesh with required resolution
        create_mesh(r1_min, r1_max, r2_min, r2_max, mesh_resolutions[i], mesh_filename)
        # Volume in r1 direction.
        V1 = 4 * np.pi * np.power(R - sigma, 3) / 3
        # Read in mesh.
        mesh = Mesh(mesh_folder + mesh_filename + ".xml")
        bdry = MeshFunction("size_t", mesh, mesh_folder + mesh_filename + "_facet_region.xml")
        hh.append(mesh.hmax())
        # Solve problem
        p_h, p_approx, flux, flux_approx = solve_problem_instance_mixed(c[0], t_final, dt, mesh, bdry, sigma, deg, steady_state)
        #E_u_H1 = assemble((grad(p_approx) - grad(p_h)) ** 2 * dx)
        #E_u_L2 = assemble((p_approx - p_h) ** 2 * dx)
        #E_u_L2 = assemble((flux_approx - flux) ** 2 * dx)
        p_steady_state = PInitial(c[0], sigma, degree= 4)
        p_flux_approx = SInitial(c[0], sigma, degree= 4)

        #eu.append(pow(E_u_H1, 0.5))
        #e0.append(pow(E_u_L2, 0.5))
        eu.append(errornorm(p_steady_state, p_h, norm_type='H10'))
        e0.append(errornorm(p_flux_approx, flux, norm_type='L2'))

        if (i > 0):
            ru.append(ln(eu[i] / eu[i - 1]) / ln(hh[i] / hh[i - 1]))
            r0.append(ln(e0[i] / e0[i - 1]) / ln(hh[i] / hh[i - 1]))
    # ********* Generating error history ****** #
    print('====================================================')
    print('  h    e_1(u)   r_1(u)   e_0(u)  r_0(u)    ')
    print('====================================================')
    for nk in range(len(mesh_resolutions)):
        print('{:.4f} {:6.2e}  {:.3f}  {:6.2e}  {:.3f} '.format(hh[nk], eu[nk], ru[nk], e0[nk], r0[nk]))
    print('====================================================')



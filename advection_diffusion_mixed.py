import numpy as np
from fenics import *

np.seterr(all='raise')
np.seterr(under='ignore')


parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4


def exp_reaction_boundary(sigma, gamma, r2_val):
    return sigma*np.exp(-4/3*np.pi*gamma*np.power(r2_val, 3))


def Gstar(p, q, c, r2):
    """
    The correct non-linear term is (r2*p)^2/q but for sufficiently large r2 this limits to 4*np.pi*c*p*r2**2.
    Using this approximation avoids the floating point errors that arise from computing (r2*p)^2/q when p and q are
    very small.
    """
    return conditional(gt(abs(1/(4*c*np.pi)*p - q), 0.001), (p/q)*p*r2**2, 4*np.pi*c*p*r2**2)


def div_rad(vec, r1, r2):
    return Dx(r2**2*vec[0],0)/r2**2 + Dx(r1**2*vec[1],1)/r1**2


class PInitial(UserExpression):
    """
    Custom expression to avoid the issues Fenics seems to have with exp().
    """
    def __init__(self, conc, sigma, gamma, V1, **kwargs):
        super().__init__(**kwargs)
        self.conc = conc
        self.sigma = sigma
        self.gamma = gamma
        self.V1 = V1

    def eval(self, values, x):
        if x[1] > 0:
            values[0] = self.conc/self.V1*np.exp(-4/3*np.pi*self.conc*np.power(x[0],3))*(1 - exp_reaction_boundary(self.sigma,self.gamma,x[0])/x[1])
        else:
            values[0] = self.conc / self.V1 * np.exp(-4 / 3 * np.pi * self.conc * np.power(x[0], 3))

    def value_shape(self):
        return ()


class SInitial(UserExpression):
    """
    Custom expression to avoid the issues Fenics seems to have with exp().
    """
    def __init__(self, conc, sigma, gamma, V1, **kwargs):
        super().__init__(**kwargs)
        self.conc = conc
        self.sigma = sigma
        self.gamma = gamma
        self.V1 = V1

    def eval(self, values, x):
        values[0] = 0
        if np.square(x[1]) > 0:
            try:
                values[1] = -self.conc/self.V1*np.exp(-4/3*np.pi*self.conc*np.power(x[0],3))*\
                        (exp_reaction_boundary(self.sigma, self.gamma, x[0])/np.square(x[1]))
            except:
                #print(f'r1: {x[1]}, r2: {x[0]}')
                pass
        else:
            values[1] = 0

    def value_shape(self):
        return (2,)


class QInitial(UserExpression):
    """
    Custom expression to avoid the issues Fenics seems to have with exp().
    """
    def __init__(self, conc, V1, **kwargs):
        super().__init__(**kwargs)
        self.conc = conc
        self.V1 = V1

    def eval(self, values, x):
        values[0] = 1/(4*np.pi*self.V1)*np.exp(-4/3*np.pi*self.conc*np.power(x[0], 3))

    def value_shape(self):
        return ()


def solve_problem_instance(concentration, t_final, dt, mesh, bdry, sigma, gamma, results, V1, D1, D2, output_filename,
                           steady_state = False):
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
        The mesh function that marks the boundaries of 'mesh'.
    sigma: float
        The maximum height of the bottom/reaction boundary in the mesh. We assume here the boundary is
        ~ sigma*exp(-4*pi*gamma*r2^3/3).
    gamma: float
        The rate of decay for the bottom boundary in the mesh.
    results: list[float]
        Initially empty this list is populated with the flux computations (namely the net flux over the bottom boundary)
        at the end of the computation.
    output_filename: str
        The filename to write the finite element solution to.
    steady_state: bool
        Whether to solve the steady-state problem or not.
    """
    r2, r1 = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    hK = CellDiameter(mesh)
    weight = (4 * pi * r1 * r2) ** 2
    r2_vec = Constant((1, 0))
    r1_vec = Constant((0, 1))
    r1_matrix = as_tensor([[0, 0], [0, 1]])
    r2_matrix = as_tensor([[1, 0], [0, 0]])
    D = as_tensor([[D2, 0], [0, D1]])
    stab = Constant(0.001)
    deg = 2
    beta = 1
    alpha = 7/2
    stab_factor = stab / (deg + 1) ** alpha * avg(hK) ** 2 * beta

    # mesh labels
    right = 21
    top = 22
    left = 23
    bottom = 24

    # ******** Finite dimensional spaces ******** #
    P0 = FiniteElement('DG', mesh.ufl_cell(), deg)
    RT1 = FiniteElement('RT', mesh.ufl_cell(), deg + 1)
    P1 = FiniteElement('CG', mesh.ufl_cell(), deg + 1)
    mixed_space = FunctionSpace(mesh, MixedElement([P0, RT1, P1]))

    # ******** Initialise output file ******** #
    output_file = XDMFFile(output_filename)
    output_file.parameters['rewrite_function_mesh'] = False
    output_file.parameters["flush_output"] = True
    output_file.parameters["functions_share_mesh"] = True
    # ******** Initialise time evolution ******** #
    if steady_state:
        t_final = 0
    t = 0
    # ********* test and trial functions ******** #
    v, tau, w = TestFunctions(mixed_space)
    u = Function(mixed_space)
    du = TrialFunction(mixed_space)
    p, s, q = split(u)

    # ********* initial and boundary conditions ******** #
    # ********* initial and boundary conditions ******** #
    p_initial = PInitial(concentration, sigma, gamma, V1, degree=deg)
    q_initial = QInitial(concentration, V1, degree=deg+1)

    p_old = interpolate(p_initial, mixed_space.sub(0).collapse())

    # as the formulation for p-fluxp is mixed, the boundary condition for p becomes natural and the boundary condition
    # for the flux becomes essential (dirichlet)
    p_right = p_initial
    p_top = p_initial

    # Boundary conditions for s are complementary to those of p:
    #s_left = SInitial(concentration, sigma, gamma, V1)
    s_left_boundary_condition = DirichletBC(mixed_space.sub(1), (Constant(0), Constant(0)), bdry,
                                            left)

    # the formulation for q is primal, so the Dirichlet conditions remain so
    q_right = q_initial

    q_right_boundary_condition = DirichletBC(mixed_space.sub(2), q_right, bdry, right)

    # here we only list the Dirichlet conditions
    bc = [s_left_boundary_condition, q_right_boundary_condition]
    ds = Measure('ds', domain=mesh, subdomain_data=bdry)

    # ********* Weak forms ********* #
    beta = Constant(1.)
    if not steady_state:
        # Time dependent weak form.
        FF = (p - p_old) / dt * v * weight * dx \
             + ((D1 / r1 ** 2) * Dx(r1 ** 2 * (dot(s, r1_vec)), 1) + (D2 / r2 ** 2) * Dx(r2 ** 2 * (dot(s, r2_vec)),
                                                                                         0)) * v * weight * dx \
             + dot(s + (Gstar(p, q, concentration, r2) * r2_vec), tau) * weight * dx \
             - p * ((1 / r1 ** 2) * Dx(r1 ** 2 * (dot(tau, r1_vec)), 1) + (1 / r2 ** 2) * Dx(
            r2 ** 2 * (dot(tau, r2_vec)), 0)) * weight * dx \
             + p_right * dot(tau, n) * weight * ds(right) \
             + p_top * dot(tau, n) * weight * ds(top) \
             + Dx(q, 0) * w * weight * dx + r2 ** 2 * p * w * weight * dx

        # Last term is from the boundary condition for q which states dq/dr2 = 0 on the inner boundary.
    else:
        # Steady-state weak form.
        FF = ((D1 / r1 ** 2) * Dx(r1 ** 2 * (dot(s, r1_vec)), 1) + (D2 / r2 ** 2) * Dx(r2 ** 2 * (dot(s, r2_vec)),
                                                                                         0)) * v * weight * dx \
             + dot(s + (Gstar(p, q, concentration, r2) * r2_vec), tau) * weight * dx \
             - p * ((1 / r1 ** 2) * Dx(r1 ** 2 * (dot(tau, r1_vec)), 1) + (1 / r2 ** 2) * Dx(
            r2 ** 2 * (dot(tau, r2_vec)), 0)) * weight * dx \
             + p_right * dot(tau, n) * weight * ds(right) \
             + p_top * dot(tau, n) * weight * ds(top) \
             + Dx(q, 0) * w * weight * dx + r2 ** 2 * p * w * weight * dx

    Tang = derivative(FF, u, du)
    problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs=bc)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver'] = 'newton'
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
    solver.parameters['newton_solver']['maximum_iterations'] = 10

    # (approximate) steady-state solution used for comparison
    p_steady_state = PInitial(concentration, sigma, gamma, V1, degree=deg)
    p_flux_approx = SInitial(concentration, sigma, gamma, V1, degree=deg+1)

    p_approx = 4 * np.pi * r2 ** 2 * interpolate(p_steady_state, mixed_space.sub(0).collapse())
    flux_approx = D*4 * np.pi * r2 ** 2 *interpolate(p_flux_approx, mixed_space.sub(1).collapse())
    while (t <= t_final):
        print("t=%.3f" % t)
        solver.solve()
        p_h, s_h, q_h = u.split()
        adjusted_p_h = project(4*np.pi*r2**2*conditional(gt(p_h, 0), p_h, 0), mixed_space.sub(0).collapse())
        adjusted_q_h = project(conditional(gt(q_h, 0), q_h, 0), mixed_space.sub(2).collapse())
        # Compute flux
        flux = project(D*4*np.pi*r2**2*s_h, mixed_space.sub(1).collapse())
        # Save the actual solution
        adjusted_p_h.rename("p", "p")
        q_h.rename("q", "q")
        flux.rename("flux", "flux")
        output_file.write(adjusted_p_h, t)
        output_file.write(adjusted_q_h, t)
        output_file.write(flux, t)
        # Compare solution to approximation
        difp = project((adjusted_p_h - p_approx), mixed_space.sub(0).collapse())
        difp.rename("dif", "dif")
        output_file.write(difp, t)
        # Compare flux to approximation
        dif_flux = project(flux - flux_approx, mixed_space.sub(1).collapse())
        dif_flux.rename("dif_flux", "dif_flux")
        output_file.write(dif_flux, t)
        # Update the solution for next iteration
        assign(p_old, p_h)
        t += dt
    # Try to compute flux over boundary
    total_flux = V1 * assemble(dot(flux, n) * 4 * pi * r1 ** 2 * ds(bottom))
    total_r1_flux = V1 * assemble(dot(flux, r1_matrix * n) * 4 * pi * r1 ** 2 * ds(bottom))
    total_r2_flux = V1 * assemble(dot(flux, r2_matrix * n) * 4 * pi * r1 ** 2 * ds(bottom))
    total_approx_flux = 4*np.pi*D1*sigma*(concentration/(gamma+concentration))

    print(f'r1 flux: {total_r1_flux} r2 flux: {total_r2_flux} total flux: {total_flux} approximate flux: '
          f'{total_approx_flux} error: {total_flux - total_approx_flux} sigma: {sigma} sigma sq: {sigma ** 2} expected flux: {4*np.pi*2*0.1*concentration/(concentration+1)}')
    results.append([sigma, gamma, concentration, total_r1_flux, total_r2_flux, total_flux, total_approx_flux])
    return flux, total_flux



if __name__ == '__main__':
    # Concentration of C molecules.
    c = np.arange(1, 10, 1)
    # The name of the output file for the flux results. Not the numerical solution, see solve_problem_instance() for
    # that output file.
    output_filename = 'corrections/FE_particle/particle_test_finite_correction'
    # The meshes to solve the problem for. Represented as an array to allow scanning through multiple problem instances.
    mesh_filenames = ["exp_boundary_sigma0.184_gamma2.73_r1max5_r2max5"]*len(c)
    # ******* Model constants ****** #
    # Sigma and gamma values must match the boundaries of the meshes in 'mesh_filenames'.
    sigmas = [0.1]*len(c)
    gammas = [1]*len(c)#np.arange(3.25, 5.25, 0.25)
    mesh_folder = "meshes/"
    # Dimensions of the mesh
    r1_max = [5]*len(c)
    r2_min = 0
    r2_max = [5]*len(c)
    R = r1_max
    # Diffusion coefficients.
    D_A = 1
    D_B = 1
    D_C = 5
    D1 = D_A + D_B
    D2 = D_C + 1 / (1 / D_A + 1 / D_B)
    # ********** Time constants ********* #
    dt = 0.1
    t_final = 0.1
    # Toogle to solve steady state or time dependent problem
    steady_state = True
    for i in range(len(c)):
        results = []
        mesh_filename = mesh_filenames[i]
        sigma = sigmas[i]
        gamma = gammas[i]
        r1_min = sigma
        # Volume in r1 direction.
        V1 = 4 * np.pi * np.power(r1_max[i] - sigma, 3) / 3
        # Volume r2 direction (space is actually 6D not 2D).
        V2 = 4 * np.pi * np.power(r2_max[i], 3) / 3
        # Read in mesh.
        mesh = Mesh(mesh_folder + mesh_filename + ".xml")
        bdry = MeshFunction("size_t", mesh, mesh_folder + mesh_filename + "_facet_region.xml")
        # Solve problem
        finite_element_filename = f'exp_boundary_convergence_c{c[i]}_sigma{sigma:.2f}_gamma{gamma:.2f}.xdmf'
        solve_problem_instance(c[i], t_final, dt, mesh, bdry, sigma, gamma, results, V1, D1, D2, finite_element_filename, steady_state)
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
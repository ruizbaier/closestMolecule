from fenics import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
parameters['reorder_dofs_serial'] = False


# output file
output_file = XDMFFile("outputs/flat_solutions.xdmf")
output_file.parameters['rewrite_function_mesh']=False
output_file.parameters["flush_output"] = True
output_file.parameters["functions_share_mesh"] = True

mesh = Mesh("meshes/flat_boundary.xml")
bdry = MeshFunction("size_t", mesh, "meshes/flat_boundary_facet_region.xml")


r2, r1 = SpatialCoordinate(mesh)

# mesh labels
right = 21
top = 22
left = 23
bottom = 24


# ******* Model constants ****** #
c = Constant(1.0)
V = Constant(1.0)
sigma = Constant(0.05)
gamma = Constant(1.0)
D1 = 2*Constant(0.01)
D2 = 3/2*Constant(0.01)
r2_vec = Constant((1, 0))
r1_vec = Constant((0, 1))
f = Constant(0.)

# ********** Time constants ********* #
t = 0
dt = 0.01
tfinal = 0.1

# ********* Finite dimensional spaces ********* #
deg = 2
P0 = FiniteElement('DG',mesh.ufl_cell(),deg-1)
RT1 = FiniteElement('RT',mesh.ufl_cell(),deg)
P1 = FiniteElement('CG', mesh.ufl_cell(), deg)
mixed_space = FunctionSpace(mesh, MixedElement([P0, RT1, P1]))

# ********* test and trial functions ****** #
v1, tau, v2 = TestFunctions(mixed_space)
u = interpolate(Expression(("(c/V)*exp(-(4/3)*pi*c*pow(x[0],3))*(1-o/x[1])", "0",
                            "-c/V*exp(-4/3*pi*c*pow(x[0],3))*(o/pow(x[1],2))",
                            "(1/(4*pi*V))*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1])"), degree=deg, c=c, V=V, o=sigma,
                           domain=mesh), mixed_space)
du = TrialFunction(mixed_space)
p, s, q = split(u)

# ********* initial and boundary conditions ******** #
p_old = interpolate(Expression("(c/V)*exp(-(4/3)*pi*c*pow(x[0],3))*(1-o/x[1])", degree=deg, c=c, V=V,
                                           o=sigma, domain=mesh),mixed_space.sub(0).collapse())

p_right = Expression("(c/V)*exp(-(4/3)*pi*c*pow(x[0],3))*(1 - o/x[1])", degree=deg, c=c, V=V, o=sigma, domain = mesh)
p_top = Expression("(c/V)*exp(-(4/3)*pi*c*pow(x[0],3))*(1 - o/x[1])", degree=deg, c=c, V=V, o=sigma, domain = mesh)
p_bottom = Constant(0.)
    
# as the formulation for p-fluxp is mixed, the boundary condition for p becomes natural and the boundary condition for
# the flux becomes essential (dirichlet).

# the formulation for q is primal, so the Dirichlet conditions remain so
q_right = Expression("1/(4*pi*V)*exp(-(4/3)*pi*c*pow(x[0],3))*(1 - o/x[1])", degree=deg, c=c, V=V, o=sigma, domain=mesh)

q_right_boundary_condition = DirichletBC(mixed_space.sub(2), q_right, bdry, right)

#Boundary conditions for s are complementary to those of p:
# The vector components are zero in the r2 direction but not in the r1
s_left = Expression(("0", "-(c/V)*exp(-(4/3)*pi*c*pow(x[0],3))*(o/pow(x[1],2))"), degree=deg, c=c, V=V, o=sigma, domain=mesh)
s_left_boundary_condition = DirichletBC(mixed_space.sub(1), s_left, bdry, left)

# here we only list the Dirichlet ones 
bc = [s_left_boundary_condition, q_right_boundary_condition]

    
# (approximate) steady-state solutions used for comparison
p_steady_state = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1])", degree=deg-1, o=sigma,  c=c, V=V, domain=mesh)

p_approx = interpolate(p_steady_state, mixed_space.sub(0).collapse())


def Gstar(p, q):
    return conditional(gt(q, 0), (r2**2*p**2)/q, 0)


# ********* Weak forms ********* #
ds = Measure('ds', domain=mesh, subdomain_data=bdry)
n = FacetNormal(mesh)
dMatrix = as_tensor([[D2, 0], [0, D1]])
weight = (4*pi*r1*r2)**2
aux = 2*(r2_vec/r2 + r1_vec / r1)

FF = (p - p_old) / dt * v1 * weight * dx \
     + ((D1/r1**2)*Dx(r1**2*(dot(s,r1_vec)),1) + (D2/r2**2)*Dx(r2**2*(dot(s,r2_vec)),0)) * v1 * weight * dx \
     + dot(s + (Gstar(p, q)*r2_vec), tau) * weight * dx \
     - p * ((1/r1**2)*Dx(r1**2*(dot(tau,r1_vec)),1) + (1/r2**2)*Dx(r2**2*(dot(tau,r2_vec)),0)) * weight * dx \
     + p_right * dot(tau, n) * weight * ds(right) \
     + p_top * dot(tau, n) * weight * ds(top) \
     + Dx(q, 0)*v2*weight*dx + r2**2*p*v2*weight*dx

#+ p_bottom * dot(dMatrix * tau, n) * weight * ds(bottom) \
Tang = derivative(FF, u, du)
problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bc)
solver = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'newton'
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-10
solver.parameters['newton_solver']['relative_tolerance'] = 1e-10
solver.parameters['newton_solver']['maximum_iterations'] = 10


while (t <=tfinal):
    print("t=%.3f" % t)
    solver.solve()
    p_h, s_h, q_h = u.split()
    # Save the actual solution
    p_h.rename("p","p")
    q_h.rename("q","q")
    s_h.rename("s","s")
    output_file.write(p_h, t)
    output_file.write(q_h, t)
    output_file.write(s_h, t)
        
    difp = project(p_h - p_approx, mixed_space.sub(0).collapse())
    difp.rename("dif", "dif")
    output_file.write(difp, t)
    # Update the solution for next iteration
    assign(p_old, p_h)

    t += dt

total_flux = assemble(dot(dMatrix*s_h, n)*weight*ds(bottom))
print(f'Total flux: {total_flux}')
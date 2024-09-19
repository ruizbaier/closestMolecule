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

# output file
output_file = XDMFFile("outputs/flat_solutions_coupled.xdmf")
output_file.parameters['rewrite_function_mesh']=False
output_file.parameters["flush_output"] = True
output_file.parameters["functions_share_mesh"] = True


# ******* Model constants ****** #
c = 0.5
V = 1.0
sigma = 0.05
D_BASE = 0.01
D1 = 2*D_BASE
D2 = 1.5*D_BASE
r2_vec = Constant((1, 0))
r1_vec = Constant((0, 1))
f = Constant(0.)

# ********** Time constants ********* #
t = 0.
dt = 0.005
tfinal = 0.01

# mesh construction
mesh = Mesh("meshes/rect_boundary.xml")
bdry = MeshFunction("size_t", mesh, "meshes/rect_boundary_facet_region.xml")
r2, r1 = SpatialCoordinate(mesh)

# mesh labels
right = 21
top = 22
left = 23
bottom = 24

# ********* Finite dimensional spaces ********* #
deg = 1
P0 = FiniteElement('CG',mesh.ufl_cell(),deg)
P1 = FiniteElement('CG', mesh.ufl_cell(), deg)
mixed_space = FunctionSpace(mesh, MixedElement([P0, P1]))

# ********* test and trial functions ****** #
v1, v2 = TestFunctions(mixed_space)
u = Function(mixed_space)
u = interpolate(Expression(("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1 - o/x[1])",
                            "1/(4*pi*V)*(1)*exp(-4/3*pi*c*pow(x[0],3))*(1 - o/x[1])"), degree=deg, c=c, V=V, o=sigma,
                           domain=mesh), mixed_space)
du = TrialFunction(mixed_space)
p, q = split(u)

# ********* initial and boundary conditions ******** #
p_initial = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1 - o/x[1])", degree=deg, c=c, V=V, o=sigma, domain=mesh)
q_initial = Expression("1/(4*pi*V)*(1)*exp(-4/3*pi*c*pow(x[0],3))*(1 - o/x[1])", degree=deg, c=c, V=V, o=sigma,
                       domain=mesh)

p_old = interpolate(p_initial, mixed_space.sub(0).collapse())
q_old = interpolate(q_initial, mixed_space.sub(1).collapse())

# the formulation for p is primal, so the Dirichlet conditions remain so.
p_right = Expression("(c/V)*exp(-4/3*pi*c*pow(x[0],3))*(1 - o/x[1])", degree=deg, c=c, V=V, o=sigma, domain=mesh)
p_top = Expression("(c/V)*exp(-4/3*pi*c*pow(x[0],3))*(1 - o/x[1])", degree=deg, c=c, V=V, o=sigma, domain=mesh)
p_bottom = Constant(0.)

p_right_boundary_condition = DirichletBC(mixed_space.sub(0), p_right, bdry, right)
p_top_boundary_condition = DirichletBC(mixed_space.sub(0), p_top, bdry, top)
p_bottom_boundary_condition = DirichletBC(mixed_space.sub(0), p_bottom, bdry, bottom)

# the formulation for q is primal, so the Dirichlet conditions remain so
q_right = Expression("1/(4*pi*V)*(1)*exp(-4/3*pi*c*pow(x[0],3))*(1 - o/x[1])", degree=deg, c=c, V=V, o=sigma,
                     domain=mesh)

q_right_boundary_condition = DirichletBC(mixed_space.sub(1), q_right, bdry, right)#DirichletBC(mixed_space.sub(1), q_right, boundary_R)#

bc = [p_right_boundary_condition,p_top_boundary_condition,p_bottom_boundary_condition,q_right_boundary_condition]

# (approximate) steady-state solution used for comparison
p_steady_state = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1])", degree=deg, o=sigma, c=c, V=V, domain=mesh)

p_approx = interpolate(p_steady_state, mixed_space.sub(0).collapse())

def Gstar(p,q):
    return conditional(gt(q, DOLFIN_EPS), r2**2*p**2/q, 0)

# ********* Weak forms ********* #
n = FacetNormal(mesh)
dMatrix = as_tensor([[D2, 0], [0, D1]])
weight = (4*pi*r1*r2)**2

FF = (p - p_old) / dt * v1 * weight * dx \
     + dot(dMatrix*(grad(p)+Gstar(p,q)*r2_vec), grad(v1)) * weight * dx \
     + Dx(q, 0)*v2*weight*dx + r2**2*p*v2*weight*dx

#rhs  = f*v*dx
#info(NonlinearVariationalSolver.default_parameters(), 1)
#print(list_linear_solver_methods())
Tang = derivative(FF, u, du)
problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bc)
solver = NonlinearVariationalSolver(problem)
solver_type = 'newton'
solver.parameters['nonlinear_solver'] = solver_type
solver.parameters[solver_type + '_solver']['linear_solver'] = 'mumps'
solver.parameters[solver_type + '_solver']['absolute_tolerance'] = 1e-9
solver.parameters[solver_type + '_solver']['relative_tolerance'] = 1e-9
solver.parameters[solver_type + '_solver']['maximum_iterations'] = 5


while (t <=tfinal):
    print("t=%.3f" % t)
    solver.solve()
    p_h,q_h = u.split()
    # Compute flux
    flux = project(-grad(p_h), FunctionSpace(mesh, "RT", deg))
    # Save the actual solution
    p_h.rename("p","p")
    q_h.rename("q","q")
    flux.rename("flux","flux")
    output_file.write(p_h, t)
    output_file.write(q_h, t)
    output_file.write(flux, t)

    difp = project((p_h - p_approx), mixed_space.sub(0).collapse())
    difp.rename("dif","dif")
    output_file.write(difp,t)
    # Update the solution for next iteration
    assign(p_old, p_h)
    assign(q_old, q_h)

    t += dt

# Try to compute flux over boundary
ds = Measure('ds', domain=mesh, subdomain_data=bdry)
total_flux = assemble(dot(flux, n)*weight*ds(bottom))
print(f'Total flux: {total_flux}')
# plot slice of solution
r2_vals = np.arange(0, 2+0.01, 0.01)
r1_val = 0.1
dif_slice = np.zeros(len(r2_vals))
# extract solution values
actual_slice = np.zeros(len(r2_vals))
for i in range(len(r2_vals)):
    actual_slice[i] = p_h(Point(r2_vals[i], r1_val))
    dif_slice[i] = np.abs(difp(Point(r2_vals[i], r1_val)))
fig, ax1 = plt.subplots()
ax1.set_xlabel('r_2')
ax1.set_ylabel(f'Solution at r_1 = {r1_val}')
ax1.plot(r2_vals, actual_slice, color='blue', label='numerical solution')

# instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()

ax2.set_ylabel('Difference to expected solution.')
ax2.plot(r2_vals, dif_slice, color='black', label='difference')
fig.legend()

plt.show()

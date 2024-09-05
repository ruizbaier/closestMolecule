from fenics import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4


# output file
output_file = XDMFFile("outputs/flat_solutions.xdmf")
output_file.parameters['rewrite_function_mesh']=False
output_file.parameters["flush_output"] = True
output_file.parameters["functions_share_mesh"] = True

# mesh construction
mesh = Mesh("meshes/flat_boundary.xml")
bdry = MeshFunction("size_t", mesh, "meshes/flat_boundary_facet_region.xml")
r2, r1 = SpatialCoordinate(mesh)

# mesh labels
right = 21; top = 22; left = 23; bottom = 24

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
t = 0.; dt = 0.01; tfinal = 0.1




# ********* Finite dimensional spaces ********* #
deg=2
P0 = FiniteElement('DG',mesh.ufl_cell(),deg-1)
RT1 = FiniteElement('RT',mesh.ufl_cell(),deg)
P1 = FiniteElement('CG', mesh.ufl_cell(), deg)
mixed_space = FunctionSpace(mesh, MixedElement([P0, RT1, P1]))

# ********* test and trial functions ****** #
v1, tau, v2 = TestFunctions(mixed_space)
u  = Function(mixed_space)
du = TrialFunction(mixed_space)
p, s, q = split(u)

# ********* initial and boundary conditions ******** #
p_initial = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))", degree = deg, c=c, V=V, domain = mesh)

q_initial = Expression("1/(4*pi*V)*(1)*exp(-4/3*pi*c*pow(x[0],3))", degree = deg, c=c, V=V, domain = mesh)

p_old = interpolate(p_initial, mixed_space.sub(0).collapse())
q_old = interpolate(q_initial, mixed_space.sub(2).collapse())
    
p_right = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))", degree = deg, c=c, V=V, domain = mesh)
p_top = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))", degree = deg, c=c, V=V, domain = mesh)
p_bottom = Constant(0.)
    
# as the formulation for p-fluxp is mixed, the boundary condition for p becomes natural and the boundary condition for the flux becomes essential (dirichlet) 

# the formulation for q is primal, so the Dirichlet conditions remain so
q_right = Expression("1/(4*pi*V)*(1)*exp(-4/3*pi*c*pow(x[0],3))", degree = deg, c=c, V=V, domain = mesh)
q_top = Expression("1/(4*pi*V)*(1)*exp(-4/3*pi*c*pow(x[0],3))", degree = deg, c=c, V=V, domain = mesh)
q_left = Constant(1/(4*pi))
q_right_boundary_condition = DirichletBC(mixed_space.sub(2), q_right, bdry, right)
q_top_boundary_condition = DirichletBC(mixed_space.sub(2), q_top, bdry, top)
q_left_boundary_condition = DirichletBC(mixed_space.sub(2), q_left, bdry, left)

#Boundary conditions for s are complementary to those of p:
s_left_boundary_condition = DirichletBC(mixed_space.sub(1), Constant((0, 0)), bdry, left)

# here we only list the Dirichlet ones 
bc = [s_left_boundary_condition, q_right_boundary_condition,q_left_boundary_condition, q_top_boundary_condition]

    
# (approximate) steady-state solutions used for comparison
p_steady_state = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))", degree = deg, c=c, V=V, domain = mesh)

p_approx = interpolate(p_steady_state, mixed_space.sub(0).collapse())



def Gstar(p,q):
    return conditional(gt(q,0),D2*r2**2*p**2/q,0)
            
# ********* Weak forms ********* #

n = FacetNormal(mesh)
dMatrix = as_tensor([[D2,0],[0,D1]])
weight = (4*pi*r1*r2)**2
aux = 2*(r2_vec / r2 + r1_vec / r1)

FF = (p - p_old) / dt * v1 * weight * dx \
     + (div(s)+ dot(aux,s)) * v1 * weight * dx \
     + dot(s + Gstar(p,q) * r2_vec, tau) * weight * dx \
     - p * (div(dMatrix*tau) + dot(aux,dMatrix*tau)) * weight * dx \
     + p_bottom * dot(dMatrix * tau, n) * weight * ds(bottom) \
     + p_right * dot(dMatrix * tau, n) * weight * ds(right) \
     + p_top * dot(dMatrix * tau, n) * weight * ds(top) \
     + (q - q_old) / dt * v2 * weight * dx \
     + dot(dMatrix*grad(q),grad(v2)) * weight * dx \
     + (D2*4./r2) * Dx(q,0) * v2 * weight * dx \
     + r2 ** 2 * Gstar(p,q) * v2 * weight * dx \
     - dot(D1 * Dx(q,1) * v2 * r1_vec, n) * weight * ds(bottom)


# Last term is from the boundary condition for q which states dq/dr2 = 0 on the inner boundary.

Tang = derivative(FF,u,du)
problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bc)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver']                    = 'newton'
solver.parameters['newton_solver']['linear_solver']      = 'mumps'
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-7
solver.parameters['newton_solver']['relative_tolerance'] = 1e-7
solver.parameters['newton_solver']['maximum_iterations'] = 15


while (t <=tfinal):
    print("t=%.3f" % t)
    solver.solve()
    p_h,s_h,q_h = u.split()
    # Save the actual solution
    p_h.rename("p","p")
    q_h.rename("q","q")
    s_h.rename("s","s")
    output_file.write(p_h, t)
    output_file.write(q_h, t)
    output_file.write(s_h, t)
        
    difp = project(p_h - p_approx, mixed_space.sub(0).collapse())
    difp.rename("dif","dif")
    output_file.write(difp,t)
    # Update the solution for next iteration
    assign(p_old, p_h)
    assign(q_old, q_h)

    t += dt

'''
Particle movement is radial so each one depends only on r_i. 

The laplacian *is not* spherical laplacian (need to call it something else). The divergence in the radial coordinate r2 is well defined. 

what is the r1/r2 gradient? usual gradient

$$$$$ 

it seems no need for Jacobian: the eq in r2-r1 is already posed in those coordinates


$$$$$$$$

convergence seems to depend on balance between diffusion and reaction (I've put a smaller D)

 - r2 ->

______  ^
|     | |
|     | r1
|_____| |


- 1/r^2 d_r(r^2 d_r u) = - 1/r^2 * (2r * d_r(u) + r^2*d_rr(u)) = - [2/r*d_r(u) + d_rr(u)]

                             ibp
-[d_r(u)*(v*2/r) + d_rr(u)*v] =  u*d_r(2v/r)  + d_r(u)*d_r(v)
                              =  2/r*u*d_r(v) - 2/r^2*u*v + d_r(u)*d_r(v)

or with ibp only on lap:

-[d_r(u)*(v*2/r) + d_rr(u)*v] = -d_r(u)*(v*2/r)  + d_r(u)*d_r(v)
                              =  -2/r*d_r(u)*v + d_r(u)*d_r(v) 

'''



from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4


#def density(c):
#    return exp(-pow(r2,2))*sin(pi*r1)*exp(-pow(c,2))
                                   
fileO = XDMFFile("outputs/ComputingSols_test01.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True


# ******* Model constants ****** #

c = Constant(0.1)
V = Constant(1000.)


D2 = Constant(1.e-4)
D1 = Constant(1.e-3)

r2vec = Constant((1,0))

f = Constant(0.)


# probability. Let's simply approximate it by a constant. We can do this with an approximation but at the moment I've not been able to construct one for the actual integral. Only one for the line integral (of any nonlinear function!) but between two points. This is not what we want

density = Constant(1.)
# inserting a made-up nonlinear function 

t = 0.; dt = 0.1; tfinal = 20.;


deg=1;

mesh = Mesh("meshes/meshWithExpRealNewScalingConverted.xml")
bdry = MeshFunction("size_t", mesh, "meshes/meshWithExpRealNewScalingConverted_facet_region.xml");
r2, r1 = SpatialCoordinate(mesh)

# mesh labels
right = 21; top=22; left = 23; bottom = 24;

# ********* Finite dimensional spaces ********* #
    
Vh = FunctionSpace(mesh, 'CG', deg)
    
# ********* test and trial functions ****** #
    
v = TestFunction(Vh)
u = Function(Vh)
du = TrialFunction(Vh)

# ********* initial and boundary conditions (Essential) ******** #


uinit = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))",degree = 2, c=c, V=V, domain = mesh)

uold = Function(Vh) #interpolate(uinit,Vh)

bcUbot = DirichletBC(Vh, Constant(0.), bdry, bottom)
bcUright = DirichletBC(Vh, Constant(0.), bdry, right)
bcuTop = DirichletBC(Vh, uinit, bdry, top)
bcU = [bcUbot,bcUright,bcuTop]
    
# ********* Weak forms ********* #
lhs = (u-uold)/dt*v*dx + (D2*Dx(u,0)*Dx(v,0) + D1*Dx(u,1)*Dx(v,1))*dx \
    - D2*2./r2*Dx(u,0)*v*dx \
    - D1*2./r1*Dx(u,1)*v*dx \
    + D2*dot(4*pi*r2**2*u**2/density*r2vec,grad(v))*dx \
    - D2*2./r2*4*pi*r2**2*u**2/density*v*dx

# the second-last term comes from integration by parts of the first contribution to the r2-divergence operator. Then we are implicitly assuming that the total flux vanishes on left (this is why it does not appear in the weak formulation)


rhs  = f*v*dx 

FF = lhs - rhs
    
Tang = derivative(FF,u,du)
problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bcU)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver']                    = 'newton'
solver.parameters['newton_solver']['linear_solver']      = 'mumps'
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-7
solver.parameters['newton_solver']['relative_tolerance'] = 1e-7


while (t <=tfinal):
    print("t=%.3f" % t)
    
    solver.solve()
    u_h = u
    u_h.rename("u","u"); fileO.write(u_h,t)
    assign(uold,u_h)
    t += dt;
    

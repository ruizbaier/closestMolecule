'''
FEniCS version 2019.1.0

 - r2 ->

______  ^
|     | |
|     | r1
|_____| |


'''



from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# needed for convergence studies (with manufactured solutions)
import sympy2fenics as sf

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))


# user-defined functions 

def div_r2(vec):
    return D2*Dx(vec[0],0) + D2*2/r2*vec[0]

def div_r2r1(vec):
    return div_r2(vec) + D1*Dx(vec[1],1) + D1*2/r1*vec[1]

def Lap_r2r1(u):
    return div_r2r1(grad(u))

# ******* Exact solutions and forcing terms for error analysis ****** #

u_str = 'sin(pi*x)*sin(pi*y)'


D2 = Constant(1.e-4)
D1 = Constant(1.e-3)

r2vec = Constant((1,0))

# probability. Let's simply approximate it by a constant 
density = Constant(1.)


deg=1; nkmax = 7

hh = []; nn = []; eu = []; ru = [];
e0 = []; r0 = []

ru.append(0.0); r0.append(0.0); 


for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2,nk+1)
    mesh = RectangleMesh(Point(0,0),Point(1,1),nps,nps)
    r2, r1 = SpatialCoordinate(mesh)
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    # V (the continuous space for u) is expected to be H^1(\Omega)
    
    Vh = FunctionSpace(mesh, 'CG', deg) # Continuous Galerkin 
    nn.append(Vh.dim())
    
    # ********* test and trial functions ****** #
    
    v = TestFunction(Vh)
    u = Function(Vh)
    du = TrialFunction(Vh)
    
    # ********* instantiation of exact solutions ****** #
    
    u_ex    = Expression(str2exp(u_str), degree=7, domain=mesh)
    f_ex    = u_ex - Lap_r2r1(u_ex) - div_r2(4*pi*r2**2*u_ex**2/density*r2vec)

    # ********* boundary conditions (Essential) ******** #
    
    bcU = DirichletBC(Vh, u_ex, 'on_boundary')
    
    # ********* Weak forms ********* #
    lhs = u*v*dx + (D2*Dx(u,0)*Dx(v,0) + D1*Dx(u,1)*Dx(v,1))*dx \
        - D2*2./r2*Dx(u,0)*v*dx \
        - D1*2./r1*Dx(u,1)*v*dx \
        + D2*dot(4*pi*r2**2*u**2/density*r2vec,grad(v))*dx \
        - D2*2./r2*4*pi*r2**2*u**2/density*v*dx
        
    rhs  = f_ex*v*dx 

    FF = lhs - rhs

    '''
    Tang = derivative(FF,u,du)
    problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bcU)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'newton'
    solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-7
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-7

    solver.solve()
    '''
    solve(FF==0, u, bcs=bcU)
    u_h = u
    
    # ********* Computing errors ****** #

    E_u_H1 = assemble((grad(u_ex)-grad(u_h))**2*dx)
    E_u_L2 = assemble((u_ex-u_h)**2*dx)

    eu.append(pow(E_u_H1,0.5))
    e0.append(pow(E_u_L2,0.5))

    
    if(nk>0):
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        r0.append(ln(e0[nk]/e0[nk-1])/ln(hh[nk]/hh[nk-1]))
        

# ********* Generating error history ****** #
print('====================================================')
print('  DoF      h    e_1(u)   r_1(u)   e_0(u)  r_0(u)    ')
print('====================================================')
for nk in range(nkmax):
    print('{:6d}  {:.4f} {:6.2e}  {:.3f}  {:6.2e}  {:.3f} '.format(nn[nk], hh[nk], eu[nk], ru[nk], e0[nk], r0[nk]))
print('====================================================')

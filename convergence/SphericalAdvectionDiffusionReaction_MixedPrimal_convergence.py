'''
FEniCS version 2019.1.0

 - r2 ->

______  ^
|     | |
|     | r1
|_____| |

Strong form 


      s + grad(p) + G(u,p) = 0
            p - D*div_r(s) = f2
q + grad(q).r2vec + r2^2*p = f3

BCs: p=p_ex and q = q_ex everywhere

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

def div_rad(vec):
    return Dx(r2**2*vec[0],0)/r2**2 + Dx(r1**2*vec[1],1)/r1**2

def G(p,q):
    return D2*4*pi*r2**2*p**2/(q+1)/density*r2vec

# ******* Exact solutions and forcing terms for error analysis ****** #

p_str = 'sin(pi*x)*sin(pi*y)'
q_str = '2+cos(pi*x*y)'

D1 = Constant(1)
D2 = Constant(1.5)
D = Constant(((D2,0),(0,D1)))

r2vec = Constant((1,0))

# probability. Let's simply approximate it by a constant 
density = Constant(1.)


deg=0; nkmax = 4

hh = []; nn = [] 
ep = []; rp = []
eq = []; rq = []
es = []; rs = []


rp.append(0.0); rq.append(0.0); rs.append(0.0); 


for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2,nk+1)
    mesh = RectangleMesh(Point(0,0),Point(1,1),nps,nps)
    n = FacetNormal(mesh)
    r2, r1 = SpatialCoordinate(mesh)
    weight = (4 * pi * r1 * r2) ** 2
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    # Sig (cont space for flux) is Hdiv
    # Phi (cont space for q) is L2
    # Psi (the continuous space for q) is expected to be H^1(\Omega)
    
    P0 = FiniteElement('DG', mesh.ufl_cell(), deg)
    RT1 = FiniteElement('RT', mesh.ufl_cell(), deg+1)
    P1 = FiniteElement('CG', mesh.ufl_cell(), deg+1)

    Vh = FunctionSpace(mesh, MixedElement([RT1,P0,P1])) # product space  
    nn.append(Vh.dim())
    
    # ********* test and trial functions ****** #
    
    tau,v,w = TestFunctions(Vh)
    u = Function(Vh)
    s,p,q = split(u)
    du = TrialFunction(Vh)
    
    # ********* instantiation of exact solutions ****** #
    
    p_ex    = Expression(str2exp(p_str), degree=7, domain=mesh)
    q_ex    = Expression(str2exp(q_str), degree=7, domain=mesh)
    s_ex    = -grad(p_ex) - G(p_ex,q_ex)
    f2_ex   = p_ex + div_rad(D*s_ex)
    f3_ex   = q_ex + dot(grad(q_ex),r2vec) + r2**2*p_ex 

    # ********* boundary conditions (Essential) ******** #
    # p = p_ex (natural) and q=q_ex (essential) everywhere
    bcQ = DirichletBC(Vh.sub(2), q_ex, 'on_boundary')

    
    # ********* Weak forms ********* #
    lhs = dot(s+G(p,q),tau)*weight*dx +p*div(tau)*weight*dx - p*div_rad(tau)*weight*dx  \
            - v*div_rad(D*s)*weight*dx - p*v*weight*dx \
            + (q+dot(grad(q),r2vec)+r2**2*p)*w*weight*dx 
    
    rhs  = p_ex*dot(tau,n)*weight*ds - f2_ex*v*weight*dx + f3_ex*w*weight*dx  

    FF = lhs - rhs

    Tang = derivative(FF,u,du)
    problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bcQ)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'newton'
    solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-7
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-7

    solver.solve()
    s_h, p_h, q_h = u.split()
    
    # ********* Computing errors ****** #

    E_q_H1 = assemble((grad(q_ex)-grad(q_h))**2*dx)
    E_p_L2 = assemble((p_ex-p_h)**2*dx)
    E_s_Hdiv = assemble((s_ex-s_h)**2*dx+(div(s_ex)-div(s_h))**2*dx)

    eq.append(pow(E_q_H1,0.5))
    ep.append(pow(E_p_L2,0.5))
    es.append(pow(E_s_Hdiv,0.5))

    
    if(nk>0):
        rs.append(ln(es[nk]/es[nk-1])/ln(hh[nk]/hh[nk-1]))
        rp.append(ln(ep[nk]/ep[nk-1])/ln(hh[nk]/hh[nk-1]))
        rq.append(ln(eq[nk]/eq[nk-1])/ln(hh[nk]/hh[nk-1]))
        

# ********* Generating error history ****** #
print('====================================================')
print('  DoF      h     e_div(s)   r_div(s)   e_0(p)   r_0(p)   e_1(q)  r_1(q)    ')
print('====================================================')
for nk in range(nkmax):
    print('{:6d}  {:.4f}  {:6.2e}  {:.3f}  {:6.2e}  {:.3f}  {:6.2e}  {:.3f} '.format(nn[nk], hh[nk], es[nk], rs[nk], ep[nk], rp[nk], eq[nk], rq[nk]))
print('====================================================')

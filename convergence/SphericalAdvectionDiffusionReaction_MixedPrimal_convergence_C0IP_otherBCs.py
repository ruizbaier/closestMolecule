'''
FEniCS version 2019.1.0

 - r2 ->

______  ^
|     | |
|     | r1
|_____| |

Strong form 


      s + grad(p) + G(u,p) = 0
                div_r(D*s) = f2
q + grad(q).r2vec + r2^2*p = f3

BCs: p = p_ex on bot, right, top 
     s = s_ex on left
     q = q_ex on right 

mixed for s-p and C0IP for q (as in Burman-Ern 07)

'''

from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# needed for convergence studies (with manufactured solutions)
import sympy2fenics as sf

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

fileO = XDMFFile("outputs/convergences-C0IP.xdmf")
fileO.parameters['rewrite_function_mesh']=True
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# user-defined functions 

def div_rad(vec):
    return Dx(r2**2*vec[0],0)/r2**2 + Dx(r1**2*vec[1],1)/r1**2

def G(p,q):
    return D2*4*pi*r2**2*p**2/(q+DOLFIN_EPS)/density*Constant((1,0))

# ******* Exact solutions and forcing terms for error analysis ****** #

p_str = 'sin(pi*x)*sin(pi*y)'
q_str = '1.5+cos(pi*x*y)'

D1 = Constant(1e-3)
D2 = Constant(1.5e-3)
D = Constant(((D2,0),(0,D1)))
r2vec = Constant((-1,0))
density = Constant(1.)
stab = Constant(0.00075)

deg=0; nkmax = 7

hh = []; nn = [] 
ep = []; rp = []
eq = []; rq = []
es = []; rs = []


rp.append(0.0); rq.append(0.0); rs.append(0.0); 


for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2,nk+1)
    mesh = RectangleMesh(Point(0,0),Point(1,1),nps,nps)
    bdry = MeshFunction("size_t", mesh, 1)
    bdry.set_all(0)
    left = 30; right = 31; rest= 32
    GRight = CompiledSubDomain("near(x[0],1) && on_boundary")
    GLeft = CompiledSubDomain("near(x[0],0) && on_boundary")
    GRest = CompiledSubDomain("(near(x[1],0) || near(x[1],1)) && on_boundary")
    GLeft.mark(bdry,left); GRight.mark(bdry,right); GRest.mark(bdry,rest)
    ds = Measure("ds", subdomain_data = bdry)
    hK = CellDiameter(mesh)
    n = FacetNormal(mesh)
    r2, r1 = SpatialCoordinate(mesh)
    weight = (4 * pi * r1 * r2) ** 2
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    # Sig (cont space for flux) is Hdiv_rad
    # Phi (cont space for q) is L2_rad
    # Psi (the continuous space for q) is expected to be H^1_rad
    
    RTk  = FiniteElement('RT', mesh.ufl_cell(), deg+1)
    Pk   = FiniteElement('DG', mesh.ufl_cell(), deg)
    Pkp1 = FiniteElement('CG', mesh.ufl_cell(), deg+1)

    Vh = FunctionSpace(mesh, MixedElement([RTk,Pk,Pkp1])) 
    nn.append(Vh.dim())
    
    # ********* test and trial functions ****** #
    
    du = TrialFunction(Vh)
    u = Function(Vh)
    sig,p,q = split(u)
    tau,v,w = TestFunctions(Vh)
    
    # ********* instantiation of exact solutions ****** #
    
    p_ex    = Expression(str2exp(p_str), degree=5, domain=mesh)
    q_ex    = Expression(str2exp(q_str), degree=5, domain=mesh)
    sig_ex  = -grad(p_ex) - G(p_ex,q_ex)
    f2_ex   = div_rad(D*sig_ex)
    f3_ex   = dot(grad(q_ex),r2vec)  + r2**2*p_ex 

    # ********* boundary conditions (Essential) ******** #
    # p = p_ex (natural) on top, right, bot, s = s_ex (essential) on left, and q=q_ex (essential) on right
    sigD = project(sig_ex, Vh.sub(0).collapse())
    bcS = DirichletBC(Vh.sub(0), sigD, bdry, left)
    bcQ = DirichletBC(Vh.sub(2), q_ex, bdry, right)
    bcs = [bcS,bcQ]
    beta = Constant(1.)
    
    # ********* Weak forms ********* #
    lhs = dot(sig+G(p,q),tau)*weight*dx \
            - p*div_rad(tau)*weight*dx  \
            - v*div_rad(D*sig)*weight*dx  \
            + (- div_rad(r2vec)*q + r2**2*p)*w*weight*dx \
            - dot(r2vec,grad(w))*q*weight*dx \
            + dot(r2vec,n)*q*w*weight*ds(rest) \
            + stab/(deg+1)**3.5*avg(hK)**2*beta*dot(jump(grad(q)),n('+'))*dot(jump(grad(w)),n('+'))*weight*dS
    
    rhs  = - p_ex*dot(tau,n)*weight*ds(right) \
            - p_ex*dot(tau,n)*weight*ds(rest) \
            - f2_ex*v*weight*dx \
            + f3_ex*w*weight*dx 

    FF = lhs - rhs

    Tang = derivative(FF,u,du)
    problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bcs)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'newton'
    solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-8

    solver.solve()
    sig_h, p_h, q_h = u.split()
    
    q_h.rename("q","q"); fileO.write(q_h, 1.0*nk)
    p_h.rename("p","p"); fileO.write(p_h, 1.0*nk)
    sig_h.rename("s","s"); fileO.write(sig_h, 1.0*nk)
    # ********* Computing errors ****** #

    E_q_H1 = assemble((q_ex-q_h)**2*weight*dx + (grad(q_ex)-grad(q_h))**2*weight*dx)
    E_p_L2 = assemble((p_ex-p_h)**2*weight*dx)
    E_s_Hdiv = assemble((sig_ex-sig_h)**2*weight*dx+(div_rad(sig_ex-sig_h))**2*weight*dx)

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
    print('{:6d} & {:.4f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} '.format(nn[nk], hh[nk], es[nk], rs[nk], ep[nk], rp[nk], eq[nk], rq[nk]))
print('====================================================')

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


There is an issue with setting up a Dirichlet condition on the "outflow boundary" because the upstream mechanism does not work. Burman-He 2019 https://arxiv.org/pdf/1811.00825 (sect 6.1) proposes to impose the Dirichlet on the outflow weakly. This can be done by adding the terms

LHS += (h*min(0,rvec.n)^2 + const/h)*q*w * ds
RHS += (h*min(0,rvec.n)^2 + const/h)*q_ex*w * ds

'''

from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 6

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
    return r2**2*p**2/(q+DOLFIN_EPS)*r2vec


def neg(x):
    return 0.5*(abs(x)-x)
# ******* Exact solutions and forcing terms for error analysis ****** #

p_str = 'sin(pi*x)*sin(pi*y)'
q_str = '1.5+cos(pi*x*y)'

D1 = Constant(1)
D2 = Constant(1.5)
D = Constant(((D2,0),(0,D1)))
r2vec = Constant((1,0))
deg=0; nkmax = 7

stab = Constant(0.05) #0.05 for k =0 // 0.1 for k=1
stab2= Constant(1.0*(deg+1)) #1 for k = 0 //  2 for k=1



hh = []; nn = [] 
ep = []; rp = []
eq = []; rq = []
es = []; rs = []


rp.append(0.0); rq.append(0.0); rs.append(0.0); 


for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    
    nps = pow(2,nk+1)
    mesh = UnitSquareMesh(nps,nps)
    bdry = MeshFunction("size_t", mesh, 1)
    bdry.set_all(0)
    left = 30; right = 31; rest= 32
    GLeft = CompiledSubDomain("near(x[0],0) && on_boundary")
    GRight = CompiledSubDomain("near(x[0],1) && on_boundary")
    GRest = CompiledSubDomain("(near(x[1],0) || near(x[1],1)) && on_boundary")
    GLeft.mark(bdry,left); GRight.mark(bdry,right); GRest.mark(bdry,rest)
    ds = Measure("ds", subdomain_data = bdry)
    hK = FacetArea(mesh)#CellDiameter(mesh)
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
    
    p_ex    = Expression(str2exp(p_str), degree=6, domain=mesh)
    q_ex    = Expression(str2exp(q_str), degree=5, domain=mesh)
    sig_ex  = -grad(p_ex) - G(p_ex,q_ex)
    f2_ex   = div_rad(D*sig_ex)
    f3_ex   = dot(grad(q_ex),r2vec)  + r2**2*p_ex 

    # ********* boundary conditions (Essential) ******** #
    # p = p_ex (natural) on {top, right, bot}, s = s_ex (essential) on left, and q=q_ex (weakly imposed!) on right
    sigD = project(sig_ex, Vh.sub(0).collapse())
    bcS = DirichletBC(Vh.sub(0), sigD, bdry, left)
    beta = sqrt(dot(r2vec,r2vec))
    
    # ********* Weak forms ********* #
    lhs = dot(sig+G(p,q),tau)*weight*dx \
            - p*div_rad(tau)*weight*dx  \
            - v*div_rad(D*sig)*weight*dx  \
            + (-div_rad(r2vec)*q + r2**2*p)*w*weight*dx \
            - dot(r2vec,grad(w))*q*weight*dx \
            + dot(r2vec,n)*q*w*weight*ds(left) \
            + dot(r2vec,n)*q*w*weight*ds(rest) \
            + stab/(deg+1)**3.5*avg(hK)**2*beta*dot(jump(grad(q)),n('+'))*dot(jump(grad(w)),n('+'))*weight*dS \
            + stab2*(hK*neg(dot(r2vec,n))**2+8./hK)*q*w*ds#(left)
           

    rhs  = - p_ex*dot(tau,n)*weight*ds(right) \
           - p_ex*dot(tau,n)*weight*ds(rest) \
           - f2_ex*v*weight*dx \
           + f3_ex*w*weight*dx \
           - dot(r2vec,n)*q_ex*w*weight*ds(right) \
           + stab2*(hK*neg(dot(r2vec,n))**2+8./hK)*q_ex*w*ds#(left)
           

    FF = lhs - rhs

    Tang = derivative(FF,u,du)
    problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bcS)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'snes'
    solver.parameters['snes_solver']['linear_solver']      = 'mumps'
    solver.parameters['snes_solver']['absolute_tolerance'] = 1e-7
    solver.parameters['snes_solver']['relative_tolerance'] = 1e-7
    #solver.parameters['newton_solver']['relaxation_parameter'] = 0.5
    #solver.parameters['snes_solver']['maximum_iterations'] = 100
    solver.solve()
    sig_h, p_h, q_h = u.split()
    
    q_h.rename("q","q"); fileO.write(q_h, 1.0*nk)
    p_h.rename("p","p"); fileO.write(p_h, 1.0*nk)
    sig_h.rename("s","s"); fileO.write(sig_h, 1.0*nk)
    # ********* Computing errors ****** #

    E_q_H1 = assemble((q_ex-q_h)**2*weight*dx + dot(r2vec,grad(q_ex)-grad(q_h))**2*weight*dx)
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
print('========================================================================')
print('  DoF      h     e_div(s)   r_div(s)   e_0(p)   r_0(p)   e_1(q)  r_1(q)    ')
print('========================================================================')
for nk in range(nkmax):
    print('{:6d} & {:.4f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} '.format(nn[nk], hh[nk], es[nk], rs[nk], ep[nk], rp[nk], eq[nk], rq[nk]))
print('========================================================================')


'''
k=0
   33 & 0.7071 & 2.93e+01 & 0.000 & 1.03e+00 & 0.000 & 4.92e+00 & 0.000 
   113 & 0.3536 & 1.73e+01 & 0.763 & 5.56e-01 & 0.888 & 7.08e+00 & -0.526 
   417 & 0.1768 & 9.96e+00 & 0.793 & 2.84e-01 & 0.969 & 1.84e+00 & 1.943 
  1601 & 0.0884 & 5.93e+00 & 0.747 & 1.43e-01 & 0.992 & 7.10e-01 & 1.375 
  6273 & 0.0442 & 3.72e+00 & 0.673 & 7.16e-02 & 0.997 & 3.35e-01 & 1.085 
 24833 & 0.0221 & 2.45e+00 & 0.604 & 3.59e-02 & 0.995 & 1.63e-01 & 1.036 
 98817 & 0.0110 & 1.67e+00 & 0.551 & 1.82e-02 & 0.981 & 8.13e-02 & 1.004 

k=1

    97 & 0.7071 & 1.25e+01 & 0.000 & 1.92e-01 & 0.000 & 3.60e+00 & 0.000 
   353 & 0.3536 & 4.35e+00 & 1.522 & 7.00e-02 & 1.456 & 1.10e+00 & 1.717 
  1345 & 0.1768 & 1.41e+00 & 1.625 & 1.86e-02 & 1.916 & 1.81e-01 & 2.602 
  5249 & 0.0884 & 4.79e-01 & 1.557 & 4.70e-03 & 1.981 & 4.79e-02 & 1.914 
 20737 & 0.0442 & 1.74e-01 & 1.457 & 1.18e-03 & 1.995 & 1.26e-02 & 1.929 
 82433 & 0.0221 & 6.84e-02 & 1.350 & 2.95e-04 & 1.999 & 3.21e-03 & 1.972 
328705 & 0.0110 & 2.89e-02 & 1.245 & 7.38e-05 & 2.000 & 8.07e-04 & 1.990 

   97 & 0.7071 & 1.25e+01 & 0.000 & 1.92e-01 & 0.000 & 1.84e+00 & 0.000 
   353 & 0.3536 & 4.35e+00 & 1.520 & 7.04e-02 & 1.448 & 4.16e+00 & -1.177 
  1345 & 0.1768 & 1.41e+00 & 1.627 & 1.86e-02 & 1.923 & 6.43e-01 & 2.695 
  5249 & 0.0884 & 4.79e-01 & 1.557 & 4.70e-03 & 1.982 & 1.05e-01 & 2.617 
 20737 & 0.0442 & 1.74e-01 & 1.457 & 1.18e-03 & 1.995 & 2.48e-02 & 2.081 
 82433 & 0.0221 & 6.84e-02 & 1.350 & 2.95e-04 & 1.999 & 4.67e-03 & 2.408 
328705 & 0.0110 & 2.89e-02 & 1.245 & 7.38e-05 & 2.000 & 7.75e-04 & 2.590 

'''

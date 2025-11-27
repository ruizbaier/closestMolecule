'''
FEniCS version 2019.1.0

 - r2 ->

______  ^
|     | |
|     | r1
|_____| |

Strong form 


alpha*q + grad(q).r2vec + r2^2*q = f3

BCs: q = q_ex only on the outlet!!!!!



THE C0IP formulation is from Burman-Ern Math Comp 2007

'''

from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# needed for convergence studies (with manufactured solutions)
import sympy2fenics as sf

def div_rad(vec):
    return Dx(r2**2*vec[0],0)/r2**2 + Dx(r1**2*vec[1],1)/r1**2

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

#fileO = XDMFFile("outputs/convergences.xdmf")
#fileO.parameters['rewrite_function_mesh']=True
#fileO.parameters["functions_share_mesh"] = True
#fileO.parameters["flush_output"] = True

# user-defined functions 

# ******* Exact solutions and forcing terms for error analysis ****** #
p_str = 'sin(pi*x)*sin(pi*y)'
q_str = '1.5+cos(pi*x*y)'

D1 = Constant(1)
D2 = Constant(1.5)
D = Constant(((D2,0),(0,D1)))
r2vec = Constant((1,0))
density = Constant(1.)
alpha = Constant(1.5)
stab = Constant(0.05)

#deg=2; 
nkmax = 6

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
    inlet = 31; outlet= 32; rest = 33
    Gin  = CompiledSubDomain("near(x[0],0) && on_boundary")
    Gout = CompiledSubDomain("near(x[0],1) && on_boundary")
    Grest = CompiledSubDomain("(near(x[1],0) || near(x[1],1)) && on_boundary")

    Gin.mark(bdry,inlet); Gout.mark(bdry,outlet); Grest.mark(bdry,rest)
    ds = Measure("ds", subdomain_data = bdry)
    he = CellDiameter(mesh)
    n = FacetNormal(mesh)
    r2, r1 = SpatialCoordinate(mesh)
    
    hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #

    deg = 1
    Vh = FunctionSpace(mesh, 'CG', deg+1) 
    nn.append(Vh.dim())
    
    # ********* test and trial functions ****** #
    
    q = TrialFunction(Vh)
    w = TestFunction(Vh)
    
    # ********* instantiation of exact solutions ****** #
    q_ex    = Expression(str2exp(q_str), degree=5, domain=mesh)
    f3_ex   =  dot(grad(q_ex),r2vec) 

    # ********* boundary conditions (Essential) ******** #


    # ********* Weak forms ********* #

    nitsch =Constant(1.)
    
    lhs = -Constant(0.)*q*w*dx \
        - dot(r2vec,grad(w))*q*dx \
        + dot(r2vec,n)*q*w*ds(inlet) \
        + dot(r2vec,n)*q*w*ds(rest) \
        + stab/(deg+1)**2*avg(he)**2*abs(dot(r2vec,n('+')))*dot(jump(grad(q)),n('+'))*dot(jump(grad(w)),n('+'))*dS \
        + nitsch/he*q*w*ds(outlet)
    
    rhs  = f3_ex*w*dx \
        - dot(r2vec,n)*q_ex*w*ds(outlet) \
        + nitsch/he*q_ex*w*ds(outlet)

    q_h = Function(Vh)
    solve(lhs==rhs,q_h)

    # ********* Computing errors ****** #

    E_q_H1 = assemble((q_ex-q_h)**2*dx + (grad(q_ex)-grad(q_h))**2*dx)

    eq.append(pow(E_q_H1,0.5))

    
    if(nk>0):
        
        rq.append(ln(eq[nk]/eq[nk-1])/ln(hh[nk]/hh[nk-1]))
        

# ********* Generating error history ****** #
print('====================================================')
print('  DoF      h     e_div(s)   r_div(s)   e_0(p)   r_0(p)   e_1(q)  r_1(q)    ')
print('====================================================')
for nk in range(nkmax):
    print('{:6d} & {:.4f} & {:6.2e} & {:.3f} '.format(nn[nk], hh[nk], eq[nk], rq[nk]))
print('====================================================')

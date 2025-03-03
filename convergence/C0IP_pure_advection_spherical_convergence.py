'''
FEniCS version 2019.1.0

 - r2 ->

______  ^
|     | |
|     | r1
|_____| |

Strong form 


alpha*q + grad(q).r2vec + r2^2*q = f3

BCs: q = q_ex only on the inlet

THE C0IP formulation is from Burman-Ern Math Comp 2007

'''

from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# needed for convergence studies (with manufactured solutions)
import sympy2fenics as sf

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

#fileO = XDMFFile("outputs/convergences.xdmf")
#fileO.parameters['rewrite_function_mesh']=True
#fileO.parameters["functions_share_mesh"] = True
#fileO.parameters["flush_output"] = True

# user-defined functions 

def div_rad(vec):
    return Dx(r2**2*vec[0],0)/r2**2 + Dx(r1**2*vec[1],1)/r1**2


# ******* Exact solutions and forcing terms for error analysis ****** #

q_str = '1.5+cos(pi*x*y)'

D1 = Constant(1e-3)
D2 = Constant(1.5e-3)
D = Constant(((D2,0),(0,D1)))
r2vec = Constant((1,0))
density = Constant(1.)
stab = Constant(0.005)

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
    inlet = 31; outlet= 32; char = 33
    Gin  = CompiledSubDomain("near(x[0],0) && on_boundary")
    Gout = CompiledSubDomain("near(x[0],1) && on_boundary")
    Gchar = CompiledSubDomain("(near(x[1],0) || near(x[1],1)) && on_boundary")

    Gin.mark(bdry,inlet); Gout.mark(bdry,outlet); Gchar.mark(bdry,char)
    ds = Measure("ds", subdomain_data = bdry)
    he = CellDiameter(mesh)
    n = FacetNormal(mesh)
    r2, r1 = SpatialCoordinate(mesh)
    weight = (4 * pi * r1 * r2) ** 2
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
    f3_ex   = q_ex + dot(grad(q_ex),r2vec)

    # ********* boundary conditions (Essential) ******** #

    bcQ = DirichletBC(Vh, q_ex, bdry, inlet)

    # ********* Weak forms ********* #
    beta = Constant(1)#sqrt(abs(dot(r2vec,n('+'))))
    
    lhs = (q - div_rad(r2vec)*q)*w*weight*dx \
        - dot(r2vec,grad(w))*q*weight*dx \
        + dot(r2vec,n)*q*w*weight*ds(outlet) \
        + dot(r2vec,n)*q*w*weight*ds(char) \
        + stab/(deg+1)**3.5*avg(he)**2*beta*dot(jump(grad(q)),n('+'))*dot(jump(grad(w)),n('+'))*weight*dS
    
    rhs  = f3_ex*w*weight*dx

    q_h = Function(Vh)
    solve(lhs==rhs,q_h, bcs=bcQ)

    # ********* Computing errors ****** #

    E_q_H1 = assemble((q_ex-q_h)**2*weight*dx + (grad(q_ex)-grad(q_h))**2*weight*dx)

    eq.append(pow(E_q_H1,0.5))

    
    if(nk>0):
        
        rq.append(ln(eq[nk]/eq[nk-1])/ln(hh[nk]/hh[nk-1]))
        

# ********* Generating error history ****** #
print('====================================================')
print('  DoF      h      e_1(q)  r_1(q)    ')
print('====================================================')
for nk in range(nkmax):
    print('{:6d} & {:.4f} & {:6.2e} & {:.3f} '.format(nn[nk], hh[nk], eq[nk], rq[nk]))
print('====================================================')

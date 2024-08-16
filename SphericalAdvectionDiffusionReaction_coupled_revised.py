'''
 - r2 ->

______  ^
|     | |
|     | r1
|_____| |

'''


import numpy as np
from fenics import *
import contextlib

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4


fileO = XDMFFile("outputs/all_sols.xdmf")
fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["flush_output"] = True
fileO.parameters["functions_share_mesh"] = True


# ******* Model constants ****** #

c = Constant(1.) 
V = Constant(1.)
sigma = Constant(0.05)
gamma = Constant(1.0)


D2 = Constant(0.001)
D1 = Constant(0.001)

r2vec = Constant((1,0))
r1vec = Constant((0,1))

f = Constant(0.)

t = 0.
dt = 0.01
tfinal = 0.01


deg=1

mesh = Mesh("meshes/expBoundary.xml")
bdry = MeshFunction("size_t", mesh, "meshes/expBoundary_facet_region.xml")
r2, r1 = SpatialCoordinate(mesh)

# mesh labels
right = 21
top = 22
left = 23
bottom = 24

# ********* Finite dimensional spaces ********* #
# spaces for p,s,q
P0 = FiniteElement('DG',mesh.ufl_cell(),deg-1)
RT1 = FiniteElement('RT',mesh.ufl_cell(),deg)
P1 = FiniteElement('CG', mesh.ufl_cell(), deg)
    
mixedSpace = FunctionSpace(mesh,MixedElement([P0,RT1,P1]))#Vh*gradSpace*Vh

# ********* test and trial functions ****** #
v1, tau, v2 = TestFunctions(mixedSpace)

u = Function(mixedSpace)
du = TrialFunction(mixedSpace)
p, s, q = u.split()

# ********* initial and boundary conditions ******** #
pinit = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1]*exp(-4/3*pi*g*pow(x[0],3)))",degree = 1, c=c, V=V, o = sigma, g = gamma, domain = mesh)

qinit = Expression("(1/(4*pi*x[1]*V*(c+g)))*exp(-4/3*pi*pow(x[0],3)*(c+g))*(exp(4/3*pi*pow(x[0],3)*g)*x[1]*(c+g)-c*o)",degree = 1, c=c, V=V, o = sigma, g = gamma, domain = mesh)

pOld = interpolate(pinit,mixedSpace.sub(0).collapse())
qOld = interpolate(qinit,mixedSpace.sub(0).collapse())
    
pRight = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o*exp(-4/3*pi*g*pow(x[0],3))/x[1])",degree = 2, c=c, V=V, o = sigma,g = gamma, domain = mesh)
pTop = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o*exp(-4/3*pi*g*pow(x[0],3))/x[1])",degree = 2, c=c, V=V, o = sigma,g = gamma, domain = mesh)
pBot = Constant(0.)
    
# as the formulation for p-fluxp is mixed, the boundary condition for p becomes natural and the boundary condition for the flux becomes essential (dirichlet) 

# the formulation for q is primal, so the Dirichlet condiitons remain so
qRight = Expression('(1/(4*pi*x[1]*V*(c+g)))*exp(-4/3*pi*pow(x[0],3)*(c+g))*(x[1]*(c+g)*exp(4/3*pi*pow(x[0],3)*g)-c*o)',degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)
qTop = Expression('(1/(4*pi*x[1]*V*(c+g)))*exp(-4/3*pi*pow(x[0],3)*(c+g))*(x[1]*(c+g)*exp(4/3*pi*pow(x[0],3)*g)-c*o)',degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)
bcQRight = DirichletBC(mixedSpace.sub(2), qRight, bdry, right)
bcQTop = DirichletBC(mixedSpace.sub(2), qTop, bdry, top)# QTOP??

#Boundary conditions for s are complementary to those of p:
bcSLeft = DirichletBC(mixedSpace.sub(1), Constant((0,0)), bdry, left)

# here we only list the Dirichlet ones 
bc = [bcSLeft,bcQRight,bcQTop]

    
# (approximate) steady-state solutions used for comparison
pSteady = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1]*exp(-4/3*pi*g*pow(x[0],3)))",degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)
qSteady = Expression("c/(4*pi*V)*exp(-4/3*pi*c*pow(x[0],3))",degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)

pApprox = interpolate(pSteady,mixedSpace.sub(0).collapse())
qApprox = interpolate(qSteady,mixedSpace.sub(2).collapse())


# ***** Defines nonlinear term ***** #
class Nonlocal(UserExpression):
    def __init__(self,u,**kwargs):
        super().__init__(**kwargs)
        print(p)
        self.p, self.grad ,self.q = u.split()

    def eval_cell(self,value,x,cell):
        if self.q(Point(x[0],x[1])) != 0:
            res = (D2*x[0]**2*(self.p(Point(x[0],x[1])))**2)/self.q(Point(x[0],x[1]))
            value[0] = res
        else:
            value[0] = 0

            
def G(u):
    return Constant(1.)# Nonlocal(u)


# ********* Weak forms ********* #

# not sure how to implement the boundary term on the 3rd line of the weak form (Eq24 in write up) PENDING 
n = FacetNormal(mesh)

# Matrix for diffusion coefficients (think they'll need to be backwards since r2 is the first coordinate)
dMatrix = as_matrix([[D2,0],[0,D1]])

weight = (4*pi*r1*r2)**2

lhs = (p-pOld)/dt*v1*weight*dx \
    + (div(s)+ dot(as_vector((2/r2,2/r1)),s))*v1*weight*dx \
    + dot(s + D2*G(u)*r2vec,tau)*weight*dx \
    - p*(div(dMatrix*tau) + dot(as_vector((2/r2,2/r1)),tau))*weight*dx \
    + pBot*dot(dMatrix*tau,n)*weight*ds(bottom)\
    + pRight*dot(dMatrix*tau,n)*weight*ds(right)\
    + pTop*dot(dMatrix*tau,n)*weight*ds(top) \
    + (q-qOld)/dt*v2*weight*dx \
    + dot(dMatrix*grad(q),grad(v2))*weight*dx \
    + D2*4./r2*Dx(q,0)*v2*weight*dx \
    + r2**2*G(u)*v2*weight*dx\
    - dot(D1*Dx(q,1)*v2*r1vec,n)*weight*ds(bottom)

# CHECK 3rd from bottom!

# Last term is from the boundary condition for q which states dq/dr2 = 0 on the inner boundary. CHECK IF it needs the weight!

FF = lhs

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
    #with contextlib.redirect_stdout(None):
    solver.solve()
    p_h,s_h,q_h = u.split()
    # Save the actual solution
    p_h.rename("p","p")
    q_h.rename("q","q")
    s_h.rename("s","s")
    fileO.write(p_h,t)
    fileO.write(q_h,t)
    fileO.write(s_h,t)
        
    dif = Function(mixedSpace)
    difp, difs, difq = dif.split()
    difp.vector()[:] = (p_h.vector()-pApprox.vector())
    difp.rename("dif","dif")
    fileO.write(difp,t)
    # Update the solution for next iteration
    assign(pOld,p); assign(qOld,q)

    t += dt

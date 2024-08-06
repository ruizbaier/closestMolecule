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


cVals = [1]
for c in cVals:

    fileSol = XDMFFile("outputs/expSolMixed_c"+str(c)+".xdmf")
    fileSol.parameters["functions_share_mesh"] = True
    fileSol.parameters["flush_output"] = True

    fileFlux = XDMFFile("outputs/expFluxMixed_c"+str(c)+".xdmf")
    fileFlux.parameters["functions_share_mesh"] = True
    fileFlux.parameters["flush_output"] = True

    fileApprox = XDMFFile("outputs/expSolMixed_Approx_c"+str(c)+".xdmf")
    fileApprox.parameters["functions_share_mesh"] = True
    fileApprox.parameters["flush_output"] = True


    # ******* Model constants ****** #

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


    deg=2

    mesh = Mesh("meshes/expBoundary.xml")
    bdry = MeshFunction("size_t", mesh, "meshes/expBoundary_facet_region.xml")
    r2, r1 = SpatialCoordinate(mesh)

    # mesh labels
    right = 21
    top = 22
    left = 23
    bottom = 24

    # ********* Finite dimensional spaces ********* #
    P1 = FiniteElement('CG', triangle, deg)
    # This is the space for P and Q
    Vh = FunctionSpace(mesh, P1)
    # The space for grad P
    gradSpace = FunctionSpace(mesh, 'CG', deg-1)
    # Space for the solutions (we have P, grad P and Q)
    # TODO: The fenics tutorials suggest defining the mixed space like this but it doesn't work for me so this line likely needs fixing.
    mixedSpace = Vh*gradSpace*Vh

    # ********* test and trial functions ****** #
    # A test function for each equation
    v1, tau, v2 = TestFunctions(mixedSpace)
    # Solution for each equation
    u = Function(mixedSpace)
    du = TrialFunction(mixedSpace)
    p, s, q = u.split()



    # ********* initial and boundary conditions ******** #


    # Initial condition (oldSoln will be updated each time step)
    oldSoln = Function(mixedSpace)
    # Flat boundary
    #u_0 = Expression(("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1])",'(1/(4*pi*x[1]*V))*exp(-4/3*pi*pow(x[0],3)*c)*(x[1]-o)'), degree = 2, c=c, V=V, o = sigma, domain = mesh)
    # Exp boundary
    initFunc = Expression(("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1]*exp(-4/3*pi*g*pow(x[0],3)))","0","(1/(4*pi*x[1]*V*(c+g)))*exp(-4/3*pi*pow(x[0],3)*(c+g))*(exp(4/3*pi*pow(x[0],3)*g)*x[1]*(c+g)-c*o)"), degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)
    # Has to be done this way as interpolate doesn't allow subspaces as an argument
    oldSoln = interpolate(initFunc, mixedSpace)
    pOld, grad0, qOld = oldSoln.split()



    # Conditions on p
    # Flat boundary
    #pRight = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1])",degree = 2, c=c, V=V, o = sigma, domain = mesh)
    # exp boundary
    pRight = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o*exp(-4/3*pi*g*pow(x[0],3))/x[1])",degree = 2, c=c, V=V, o = sigma,g = gamma, domain = mesh)
    pTop = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o*exp(-4/3*pi*g*pow(x[0],3))/x[1])",degree = 2, c=c, V=V, o = sigma,g = gamma, domain = mesh)
    bcPUbot = DirichletBC(mixedSpace.sub(0), Constant(0.), bdry, bottom)
    bcPUright = DirichletBC(mixedSpace.sub(0), pRight, bdry, right)
    bcuPTop = DirichletBC(mixedSpace.sub(0), pRight, bdry, top)

    # Conditions on q
    # Flat boundary
    # qRight = Expression('(1/(4*pi*x[1]*V))*exp(-4/3*pi*pow(x[0],3)*c)*(x[1]-o)',degree = 2, c=c, V=V, o = sigma, domain = mesh)
    # exp boundary
    qRight = Expression('(1/(4*pi*x[1]*V*(c+g)))*exp(-4/3*pi*pow(x[0],3)*(c+g))*(x[1]*(c+g)*exp(4/3*pi*pow(x[0],3)*g)-c*o)',degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)
    qTop = Expression('(1/(4*pi*x[1]*V*(c+g)))*exp(-4/3*pi*pow(x[0],3)*(c+g))*(x[1]*(c+g)*exp(4/3*pi*pow(x[0],3)*g)-c*o)',degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)
    bcQUright = DirichletBC(mixedSpace.sub(2), qRight, bdry, right)
    bcuQTop = DirichletBC(mixedSpace.sub(2), qRight, bdry, top)

    bcU = [bcPUbot,bcPUright,bcuPTop,bcQUright,bcuQTop]

    #TODO Boundary conditions for s (grad P)???


    # The (approximate) steady-state solutions used for comparison
    uApproxF = Expression(("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1]*exp(-4/3*pi*g*pow(x[0],3)))","0","c/(4*pi*V)*exp(-4/3*pi*c*pow(x[0],3))"),degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)#
    uApprox = interpolate(uApproxF,mixedSpace)
    pApprox, gradApprox, qApprox = uApprox.split()

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
        return Nonlocal(u)


    # ********* Weak forms ********* #

    # TODO check these are correct. Note I'm not sure how to implement the boundary term on the 3rd line of the weak form (Eq24 in write up)
    # Normal to the surface
    n = FacetNormal(mesh)
    # Matrix for diffusion coefficients (think they'll need to be backwards since r2 is the first coordinate)
    dMatrix = as_matrix([[D2,0],[0,D1]])

    # Note the gradient for the radial coordinates looks just like cartesian, but the divergence does not so we
    # cannot just use div(.) here. Also we need to include Jacobian of (4pi*r1*r2)**2 in each integration over the
    # domain, since the mesh is just 2D but our space is actually 6D.

    lhs = (p-pOld)/dt*v1*(4*np.pi*r1*r2)**2*dx + (Dx(s.sub(0),0) + Dx(s.sub(1),1) + 2.*s.sub(0)/r2 + 2.*s.sub(1)/r1)*v1*(4*np.pi*r1*r2)**2*dx\
        + dot((s + D2*G(u)*r2vec),tau) - p*(D2*Dx(tau.sub(0),0) + D1*Dx(tau.sub(1),1) + 2.*D2*tau.sub(0)/r2 + 2.*D1*tau.sub(1)/r1)*(4*np.pi*r1*r2)**2*dx\
        + p*dot(dMatrix*tau,n)*ds\
        + (q-qOld)/dt*v2*(4*np.pi*r1*r2)**2*dx + (D2*Dx(q,0)*Dx(v2,0) + D1*Dx(q,1)*Dx(v2,1))*(4*np.pi*r1*r2)**2*dx \
        + D2*4./r2*Dx(q,0)*(4*np.pi*r1*r2)**2*dx\
        + r2**2*G(u)*v2*(4*np.pi*r1*r2)**2*dx\
        - dot(D1*Dx(q,1)*v2*r1vec,n)*(4*np.pi*r1*r2)**2*ds(bottom)

    # Last term is from the boundary condition for q which states dq/dr2 = 0 on the inner boundary.

    #rhs  = f*v*dx
    FF = lhs

    Tang = derivative(FF,u,du)
    problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bcU)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']                    = 'newton'
    solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-7
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-7
    solver.parameters['newton_solver']['maximum_iterations'] = 100


    while (t <=tfinal):
        print("t=%.3f" % t)
        with contextlib.redirect_stdout(None):
            solver.solve()
        p_h,s_h,q_h = u.split()
        # Save the actual solution
        p_h.rename("p","p")
        fileSol.write(p_h,t)
        # Save the flux
        p_h.rename("s","s")
        fileFlux.write(s_h,t)
        # Save the differences between the approx and the actual solution
        dif = Function(mixedSpace)
        difp, difs, difq = dif.split()
        difp.vector()[:] = (p_h.vector()-pApprox.vector())#/p_h.vector()
        # Need to rename for correct plotting later
        difp.rename("dif","dif")
        fileApprox.write(difp,t)
        # Update the solution for next iteration
        oldSoln.assign(u)
        t += dt
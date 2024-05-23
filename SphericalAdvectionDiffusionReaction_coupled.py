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


import numpy as np
from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4





#def density(c):
#    return exp(-pow(r2,2))*sin(pi*r1)*exp(-pow(c,2))
cVals = [1]
finalFluxes = []
finalR1Fluxes = []
finalR2Fluxes = []
for c in cVals:

    fileSol = XDMFFile("outputs/ComputingSols_t0_c"+str(c)+".xdmf")
    fileSol.parameters["functions_share_mesh"] = True
    fileSol.parameters["flush_output"] = True

    fileApprox = XDMFFile("outputs/ComputingSols_Approxt0_c"+str(c)+".xdmf")
    fileApprox.parameters["functions_share_mesh"] = True
    fileApprox.parameters["flush_output"] = True

    fileFlux = XDMFFile("outputs/r1Flux_c_"+str(c)+".xdmf")
    fileFlux.parameters["functions_share_mesh"] = True
    fileFlux.parameters["flush_output"] = True


    # ******* Model constants ****** #

    V = Constant(1.)
    sigma = Constant(0.1)
    gamma = Constant(1.0)


    D2 = Constant(0.01)
    D1 = Constant(0.01)

    r2vec = Constant((1,0))
    r1vec = Constant((0,1))

    f = Constant(0.)

    # inserting a made-up nonlinear function

    t = 0.; dt = 0.005; tfinal = 0.015;


    deg=1;

    mesh = Mesh("meshes/flatBoundary.xml")
    bdry = MeshFunction("size_t", mesh, "meshes/flatBoundary_facet_region.xml");
    r2, r1 = SpatialCoordinate(mesh)

    # mesh labels
    right = 21; top=22; left = 23; bottom = 24;

    # ********* Finite dimensional spaces ********* #
    P1 = FiniteElement('CG', triangle, deg)
    # Need mixed element as the product space of our two functions
    element = MixedElement([P1, P1])
    Vh = FunctionSpace(mesh, element)
    gradSpace=FunctionSpace(mesh, 'CG', deg)

    # ********* test and trial functions ****** #
    # A test function for each equation
    v1, v2 = TestFunction(Vh)
    u = Function(Vh)
    # Access the components of the trial function
    p, q = split(u)
    du = TrialFunction(Vh)

    # ********* initial and boundary conditions (Essential) ******** #


    # Initial condition
    oldSoln = Function(Vh)
    u_0 = Expression(("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1])",'(1/(4*pi*x[1]*V))*exp(-4/3*pi*pow(x[0],3)*c)*(x[1]-o)'), degree = 2, c=c, V=V, o = sigma, domain = mesh)
    #Expression(("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1]*exp(-4/3*pi*g*pow(x[0],3)))",'(1/(4*pi*x[1]*V*(c+g)))*exp(-4/3*pi*pow(x[0],3)*(c+g))*(exp(4/3*pi*pow(x[0],3)*g)*x[1]*(c+g)-c*o)'), degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)
    oldSoln = interpolate(u_0, Vh)
    pOld, qOld = oldSoln.split()

    # Conditions on p
    pRight = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1])",degree = 2, c=c, V=V, o = sigma, domain = mesh)
    pTop = Expression('c/V*exp(-4/3*pi*c*pow(x[0],3))', degree = 2, c=c, V=V, domain = mesh)
    bcPUbot = DirichletBC(Vh.sub(0), Constant(0.), bdry, bottom)
    bcPUright = DirichletBC(Vh.sub(0), pRight, bdry, right)
    bcuPTop = DirichletBC(Vh.sub(0), pRight, bdry, top)

    # Conditions on q
    qRight = Expression('(1/(4*pi*x[1]*V))*exp(-4/3*pi*pow(x[0],3)*c)*(x[1]-o)',degree = 2, c=c, V=V, o = sigma, domain = mesh)
    qTop = Expression('c/(4*pi*V)*exp(-4/3*pi*c*pow(x[0],3))', degree = 2, c=c, V=V, domain = mesh)
    bcQUright = DirichletBC(Vh.sub(1), qRight, bdry, right)
    bcuQTop = DirichletBC(Vh.sub(1), qRight, bdry, top)

    bcU = [bcPUbot,bcPUright,bcuPTop,bcQUright,bcuQTop]

    # The initial condition tto compare to the solution
    uApproxF = Expression(("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-(o/x[1]))",'c/(4*pi*V)*exp(-4/3*pi*c*pow(x[0],3))'),degree = 2, c=c, V=V, o = sigma, domain = mesh)#
    uApprox = interpolate(uApproxF,Vh)
    pApprox, qApprox = uApprox.split()

    fluxApproxF = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(o/(pow(x[1],2)))",degree = 2, c=c, V=V, o = sigma, domain = mesh)
    fluxApprox = interpolate(fluxApproxF,gradSpace)
    # ******** Defines expression to compute the nonlocal term ********* #

    class Nonlocal(UserExpression):
        def __init__(self,u,**kwargs):
            super().__init__(**kwargs)
            print(p)
            self.p,self.q = u.split()



        def eval_cell(self,value,x,cell):
            try:
                res = (D2*x[0]**2*(self.p(Point(x[0],x[1])))**2)/self.q(Point(x[0],x[1]))
                value[0] = res
            except:
                value[0] = 0


    # ***** Defines nonlinear term ***** #
    # r2^2 is only needed here since we have factored the other r2^2 into the decomposition below
    def G(u):
        return Nonlocal(u)#(D2*r2**2*p**2)



    # ********* Weak forms ********* #
    lhs = (p-pOld)/dt*v1*dx + (D2*Dx(p,0)*Dx(v1,0) + D1*Dx(p,1)*Dx(v1,1))*dx \
        - D2*2./r2*Dx(p,0)*v1*dx \
        - D1*2./r1*Dx(p,1)*v1*dx \
        + dot(G(u)*r2vec,grad(v1))*dx \
        - (2./r2)*G(u)*v1*dx\
        + (q-qOld)/dt*v2*dx + (D2*Dx(q,0)*Dx(v2,0) + D1*Dx(q,1)*Dx(v2,1))*dx \
        - D1*2./r1*Dx(q,1)*v2*dx \
        + D2*2./r2*Dx(q,0)*v2*dx \
        + r2**2*G(u)*v2*dx\
        - Dx(q,1)*v2*ds(24)

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

    # Normal to the surface
    n = FacetNormal(mesh)
    totalFluxes = []
    r1Fluxes = []
    r2Fluxes = []
    while (t <=tfinal):
        print("t=%.3f" % t)
        solver.solve()
        p_h,q_h = u.split()
        # Save the actual solution
        p_h.rename("p","p")
        fileSol.write(p_h,t)
        totalFlux = assemble(4*pi*sigma**2*D1*4*pi*(r2**2)*dot(grad(p_h), n)*ds(24,subdomain_data=bdry))
        r1Flux = assemble(4*pi*sigma**2*D1*4*pi*(r2**2)*dot(grad(p_h), r1vec)*ds(24,subdomain_data=bdry))
        r2Flux = assemble(4*pi*sigma**2*D1*4*pi*(r2**2)*dot(grad(p_h), r2vec)*ds(24,subdomain_data=bdry))
        totalFluxes.append(totalFlux)
        r1Fluxes.append(r1Flux)
        r2Fluxes.append(r2Flux)
        print('Flux over boundary is: ' +str(totalFlux) + ' r1 flux: ' +str(r1Flux) + ' r2 flux: ' +str(r2Flux))
        # Save the differences between the approx and the actual solution
        dif = Function(Vh)
        difp, difq = dif.split()
        difp.vector()[:] = (p_h.vector()-pApprox.vector())#/p_h.vector()
        # Need to rename for correct plotting later
        difp.rename("dif","dif")
        fileApprox.write(difp,t)
        # Write the flux to a file as well
        r1FluxSol = project(Dx(p_h,1),gradSpace)
        r1FluxSol.vector()[:] =  (r1FluxSol.vector()-fluxApprox.vector())
        #r1FluxSol = r1FluxSol.sub(0)
        r1FluxSol.rename("r1FluxSol","r1_flux")
        fileFlux.write(r1FluxSol,t)
        # Update the solution for next iteration
        oldSoln.assign(u)
        t += dt
        # Compute error
        #values_u_h = u_h.compute_vertex_values(mesh)
        #values_uApprox = uApproxF.compute_vertex_values(mesh)
        #error_max = np.max(np.abs(values_u_h - values_uApprox))
        #print("max error=%.3f"%error_max)
    #print('Flux at each time step: ' +str(totalFluxes))
    finalFluxes.append(totalFluxes[-1])
    finalR1Fluxes.append(r1Fluxes[-1])
    finalR2Fluxes.append(r2Fluxes[-1])
    # Print the difference between
print('C vals: ' +str(cVals))
print('Total fluxes: ' +str(finalFluxes))
print('R1 fluxes: ' +str(finalR1Fluxes))
print('R2 fluxes: ' +str(finalR2Fluxes))



    

'''

Not sure what the spherical coordinates means here:

spherical is r, theta, phi. If we have two particles then r1,theta1,phi1,r2,theta2,phi2. But everything is written in terms of only r_i? 

Ah ok. Particle movement is radial so each one depends only on r_i. 
Anyway, the strong form of the equation is fine (it seems it's being assumed that the scalar P does not depend on anything else than r2,r1). Then the laplacian *is not* spherical laplacian (need to call it something else). The divergence in the radial coordinate r2 is well defined. 

what is the r1r2 gradient? 



But then we have a different laplacian (it is no longer spherical)

As the equation is in 2D, do we need change of coordinates (x,y -> r2,r1)? If so then we need the Jacobian when writing integral forms. If not, then we still need a Green theorem (integr by parts) in r2-r1 (not sure if it holds)

- int_Omega div_r2r1(vec) * t = int_Omega vec.grad_which?(t) - int_partialOm (vec.n)t

or do I expand the div_r2r1(vec) = div(vec) + 2/r2*vec[0] + 2/r1*vec[1], which gives (using the usual Green theorem) 

- int_Omega div_r2r1(grad(u)) * t =  - int_Omega div(grad(u)) * t  -  2/r2*dx(u) * t - 2/r1*dy(u)*t
                  = int_Omega vec.grad(t) - int_partialOm (grad(u).n)t -  2/r2*dx(u) * t - 2/r1*dy(u)*t 

are these integrals to be weighted with the Jacobian?
 

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

import sympy2fenics as sf

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

def div_r2(vec):
    return Dx(vec[0],0) + 2/r2*vec[0]

def div_r2r1(vec):
    return div(vec) + 2/r2*vec[0] + 2/r1*vec[1]

def Lap_r2r1(u):
    return div_r2r1(grad(u))

# ******* Exact solutions and forcing terms for error analysis ****** #

u_str = 'sin(pi*x)*sin(pi*y)'

#D1 = Constant(2.)
D = Constant(1e-4)

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
    
    Vh = FunctionSpace(mesh, 'CG', deg)
    nn.append(Vh.dim())
    
    # ********* test and trial functions ****** #
    
    v = TestFunction(Vh)
    u = TrialFunction(Vh)
    
    # ********* instantiation of exact solutions ****** #
    
    u_ex    = Expression(str2exp(u_str), degree=7, domain=mesh)
    f_ex    = u_ex - D*Lap_r2r1(u_ex)

    # ********* boundary conditions (Essential) ******** #
    
    bcU = DirichletBC(Vh, u_ex, 'on_boundary')
    
    # ********* Weak forms ********* #
    auv = u*v*dx + D*dot(grad(u),grad(v))*dx - D*2./r2*Dx(u,0)*v*dx - D*2./r1*Dx(u,1)*v*dx
    #auv = u*v*dx + D*dot(grad(u),grad(v))*dx + D*2./r2*Dx(v,0)*u*dx - D*2./r2**2*u*v*dx + D*2./r1*Dx(v,1)*u*dx - D*2./r1**2*u*v*dx 
    #auv = u*v*dx + D*r2*Dx(u,0)*Dx(v,0)*dx + D*r1*Dx(u,1)*Dx(v,1)*dx - D*2./r2*Dx(u,0)*v*dx - D*2./r1*Dx(u,1)*v*dx
    Fv  = f_ex*v*dx 

    u_h = Function(Vh)
    
    solve(auv == Fv, u_h, bcU) #using a direct solver
    
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

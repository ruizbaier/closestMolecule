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
                                   
fileSol = XDMFFile("outputs/ComputingSols_testBig.xdmf")
fileSol.parameters["functions_share_mesh"] = True
fileSol.parameters["flush_output"] = True

fileApprox = XDMFFile("outputs/ComputingSols_testBigApprox.xdmf")
fileApprox.parameters["functions_share_mesh"] = True
fileApprox.parameters["flush_output"] = True


# ******* Model constants ****** #

c = Constant(1.)
V = Constant(1.)
sigma = Constant(0.05)
gamma = Constant(1.0)


D2 = Constant(0.0001)
D1 = Constant(0.0001)

r2vec = Constant((1,0))

f = Constant(0.)

# inserting a made-up nonlinear function 

t = 0.; dt = 0.1; tfinal = 1.;


deg=1;

mesh = Mesh("meshes/squareMeshExpBoundarySp05FewCells.xml")
bdry = MeshFunction("size_t", mesh, "meshes/squareMeshExpBoundarySp05FewCells_facet_region.xml");
r2, r1 = SpatialCoordinate(mesh)

# mesh labels
right = 21; top=22; left = 23; bottom = 24;

# ********* Finite dimensional spaces ********* #
    
Vh = FunctionSpace(mesh, 'CG', deg)
    
# ********* test and trial functions ****** #
    
v = TestFunction(Vh)
u = Function(Vh)
du = TrialFunction(Vh)

# ********* initial and boundary conditions (Essential) ******** #


uinit = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))",degree = 2, c=c, V=V, domain = mesh)
uApproxF = Expression("c/V*exp(-4/3*pi*c*pow(x[0],3))*(1-o/x[1]*exp(-4/3*pi*g*pow(x[0],3)))",degree = 2, c=c, V=V, o = sigma, g = gamma, domain = mesh)#

uold = interpolate(uinit,Vh)
uApprox = interpolate(uApproxF,Vh)

bcUbot = DirichletBC(Vh, Constant(0.), bdry, bottom)
bcUright = DirichletBC(Vh, Constant(0.), bdry, right)
bcuTop = DirichletBC(Vh, uinit, bdry, top)
bcU = [bcUbot,bcUright,bcuTop]

# ******** Defines expressions to compute the nonlocal term ********* #
class LineIntegrand(UserExpression):
    def __init__(self,soln,**kwargs):
        super().__init__(**kwargs)
        self.soln = soln

    def eval_cell(self,value,x,cell):
        # Integrand is r^2*P(r1,r)
        value[0] = (x[0]**2)*self.soln(Point(x[0],x[1]))

class Nonlocal(UserExpression):
    def __init__(self,measures,soln,**kwargs):
        super().__init__(**kwargs)
        # The integration measures defined for each point on the mest
        self.measures = measures
        self.prevCellIndex = -1
        self.prevVal = 0
        # The expression for the integrand
        self.integrand = LineIntegrand(soln)


    def eval_cell(self,value,x,cell):
        #print(cell.index)
        # Need to select the correct measure to integrate over
        if cell.index == self.prevCellIndex:
            # Not on new cell so use previous
            value[0] = self.prevVal
        else:
            # New cell so need to reset the measure index
            self.prevCellIndex = cell.index
            val = assemble(self.integrand*self.measures[cell.index])
            if val > 10**(-5):
                value[0] = 10
                # store for next time
                self.prevVal = val
            else:
                # Below minimum for this cell
                value[0] = 10
                # store for next time
                self.prevVal = 1

marker = 1
# Need a subdomain for every cell in the domain
cellMeasures = []
'''
print(mesh.num_cells())
input()
cellCount = 0
for sourceCell in cells(mesh):
    print('Current cell: ' +str(cellCount))
    cellCount += 1
    # Print the vertex indices
    currentMeasures = []
    vertexCoordinates = [mesh.coordinates()[vertex_index] for vertex_index in sourceCell.entities(0)]
    for currentCoord in vertexCoordinates:
        subdomain_marker = MeshFunction("size_t", mesh, mesh.topology().dim())
        subdomain_marker.set_all(0)
        #print(currentCoord)
        sourceX  = currentCoord[0]
        sourceY = currentCoord[1]
        # Create new subdomain for points that contain the same x coordinate
        count = 0
        for cell in cells(mesh):
            if cell.contains(Point(sourceX,cell.midpoint().y())) and cell.midpoint().y() > sourceY:
                subdomain_marker[cell] = marker
                count+=1
        currentMeasures.append(Measure("dx", domain=mesh, subdomain_data=subdomain_marker,subdomain_id=marker))
    # Now have list of lists where we have one set of measures for each cell and the set contains a measure for each vertex in that cell
    cellMeasures.append(currentMeasures)
'''
print(mesh.num_cells())
input()
cellCount = 0
for sourceCell in cells(mesh):
    print('Current cell: ' +str(cellCount))
    cellCount += 1
    # Print the vertex indices
    subdomain_marker = MeshFunction("size_t", mesh, mesh.topology().dim())
    subdomain_marker.set_all(0)
    #print(currentCoord)
    sourceX  = sourceCell.midpoint().x()
    sourceY = sourceCell.midpoint().y()
    # Create new subdomain for points that contain the same x coordinate
    count = 0
    for cell in cells(mesh):
        if cell.contains(Point(sourceX,cell.midpoint().y())) and cell.midpoint().y() > sourceY:
            subdomain_marker[cell] = marker
            count+=1
    cellMeasures.append(Measure("dx", domain=mesh, subdomain_data=subdomain_marker,subdomain_id=marker))

# ***** Defines nonlinear term ***** #
# r2^2 is only needed here since we have factored the other r2^2 into the decomposition below
def G(u,measures):
    return D2*4*pi*r2**2*u**2/(Nonlocal(measures,u))



# ********* Weak forms ********* #
lhs = (u-uold)/dt*v*dx + (D2*Dx(u,0)*Dx(v,0) + D1*Dx(u,1)*Dx(v,1))*dx \
    - D2*2./r2*Dx(u,0)*v*dx \
    - D1*2./r1*Dx(u,1)*v*dx \
      + dot(G(u,cellMeasures)*r2vec,grad(v))*dx \
      - 2./r2*G(u,cellMeasures)*v*dx

# the second-last term comes from integration by parts of the first contribution to the r2-divergence operator.
# Then we are implicitly assuming that the total flux vanishes on left (this is why it does not appear in the weak formulation)
# Note the split of the advection into two parts comes from (D2/r2^2)*d/dr2(r2^2*G(u)) where G is given above.

rhs  = f*v*dx 
FF = lhs - rhs
    
Tang = derivative(FF,u,du)
problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs = bcU)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver']                    = 'newton'
solver.parameters['newton_solver']['linear_solver']      = 'mumps'
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-7
solver.parameters['newton_solver']['relative_tolerance'] = 1e-7

testExp = interpolate(Nonlocal(cellMeasures,uinit),Vh)
print('At (0.1,0.1): ' +str(testExp(Point(0.1,0.1))))
print('At (0.5,0.5): ' +str(testExp(Point(0.5,0.5))))
print('At (0.5,0.1): ' +str(testExp(Point(0.5,0.1))))
print('At (0.1,0.5): ' +str(testExp(Point(0.1,0.5))))
print('At (0.25,0.1): ' +str(testExp(Point(0.25,0.1))))



while (t <=tfinal):
    print("t=%.3f" % t)
    
    solver.solve()
    u_h = u
    # Save the actual solution
    u_h.rename("u","u")
    fileSol.write(u_h,t)
    # Save the differences between the approx and the actual solution
    dif = Function(Vh)
    dif.vector()[:] = u_h.vector()-uApprox.vector()
    # Need to rename for correct plotting later
    dif.rename("u","u")
    fileApprox.write(dif,t)
    # Update the solution for next iteration
    uold.assign(u_h)
    t += dt
    # Compute error
    values_u_h = u_h.compute_vertex_values(mesh)
    values_uApprox = uApproxF.compute_vertex_values(mesh)
    error_max = np.max(np.abs(values_u_h - values_uApprox))
    print("max error=%.3f"%error_max)

    

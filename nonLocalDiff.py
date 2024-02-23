from dolfin import *
import numpy as np


#################################
def line_integral(u, A, B, n): 
    '''Integrate u over segment [A, B] partitioned into n elements'''
    assert u.value_rank() == 0
    assert len(A) == len(B) > 1 and np.linalg.norm(A-B) > 0
    assert n > 0

    # Mesh line for integration
    mesh_points = [A + t*(B-A) for t in np.linspace(0, 1, n+1)]
    tdim, gdim = 1, len(A)
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 'interval', tdim, gdim) # 'triangle'?
    editor.init_vertices(n+1)
    editor.init_cells(n)

    for vi, v in enumerate(mesh_points): editor.add_vertex(vi, v)

    for ci in range(n): editor.add_cell(ci, np.array([ci, ci+1], dtype='uintp'))

    editor.close()

    # Setup funcion space
    elm = u.function_space().ufl_element()
    family = elm.family()
    degree = elm.degree()
    V = FunctionSpace(mesh, family, degree)
    v = interpolate(u, V)

    return assemble(v*dx)
############################################

mesh = UnitSquareMesh(15,15)
bdry = MeshFunction("size_t", mesh, 1); bdry.set_all(0)
Wall=CompiledSubDomain("near(x[0],0) || near(x[0],1) || near(x[1],0)")
Top=CompiledSubDomain("near(x[1],1)")
wall = 30; top = 31
Wall.mark(bdry,wall); Top.mark(bdry,top);
ds = Measure("ds", subdomain_data=bdry)



Vh = FiniteElement('RT', mesh.ufl_cell(), 1)
Qh = FiniteElement('DG', mesh.ufl_cell(), 0)
Sh = FunctionSpace(mesh,'CG',1) 
Hh = FunctionSpace(mesh, MixedElement([Vh,Qh]))

f = Expression('x[0]+x[1]', domain = mesh, degree=1)

v, q = TestFunctions(Hh)
u, p = TrialFunctions(Hh)

pold = interpolate(Constant(0),Hh.sub(1).collapse())
uold = interpolate(Constant((0,0)),Hh.sub(0).collapse())


bcu = DirichletBC(Hh.sub(0), Constant((0,0)), bdry, wall)
ptop = Constant(10.)

kappa = Constant(1.e-4)
alpha = Constant(1.e-3)

Tfinal = 1.0; dt = 0.1; t = 0.

A = np.array([0, 0])
B = np.array([0, 1])

F = Expression("1+t*(pow(u[0],2)+pow(u[1],2))*pow(p,2)", t=t, u=uold, p=pold, domain = mesh, degree=1)

lhs = 1/line_integral(project(F,Sh), A, B, n=100)*dot(u,v)*dx \
    - p*div(v)*dx \
    - q*div(u)*dx \
    - 1/dt*p*q*dx

rhs = -f*q*dx -1/dt*pold*q*dx

# ******* Solving *************** #
inc = 0
sol = Function(Hh)
while (t <= Tfinal):

    print("t=%.2f" % t)
    F.t = t; F.u = uold; F.p = pold; 
    solve(lhs == rhs, sol, bcs=bcu, solver_parameters={'linear_solver':'mumps'})

    u_h,p_h = sol.split()
    assign(uold,u_h)
    assign(pold,p_h)
    t += dt; inc += 1

    

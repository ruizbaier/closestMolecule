from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 6

import time
import sympy2fenics as sf
import os
import psutil

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# user-defined functions 

def div_rad(r1,r2,vec):
    return Dx(r2**2*vec[0],0)/r2**2 + Dx(r1**2*vec[1],1)/r1**2

def G(r2, p,q):
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
deg = 0
nkmax = 7

stab = Constant(0.05) #0.05 for k =0 // 0.1 for k=1
stab2 = Constant(1.0*(deg+1)) #1 for k = 0 //  2 for k=1

hh = []
nn = [] 
ep = []
rp = []
eq = []
rq = []
es = []
rs = []
mem = []

rp.append(0.0)
rq.append(0.0)
rs.append(0.0)

left = 30
right = 31
rest = 32
GLeft = CompiledSubDomain("near(x[0],0) && on_boundary")
GRight = CompiledSubDomain("near(x[0],1) && on_boundary")
GRest = CompiledSubDomain("(near(x[1],0) || near(x[1],1)) && on_boundary")

def run_one_mesh(nk):
    print("....... Refinement level : nk = ", nk)
    print(f"PID={os.getpid()} __name__={__name__} nk={nk}")
    
    nps = pow(2, nk+1)
    mesh = UnitSquareMesh(nps, nps)
    bdry = MeshFunction("size_t", mesh, 1)
    bdry.set_all(0)
    GLeft.mark(bdry, left)
    GRight.mark(bdry, right)
    GRest.mark(bdry, rest)
    ds = Measure("ds", subdomain_data=bdry)
    hK = FacetArea(mesh)  # CellDiameter(mesh)
    n = FacetNormal(mesh)
    r2, r1 = SpatialCoordinate(mesh)
    weight = (4 * pi * r1 * r2) ** 2
    hh.append(mesh.hmax())

    # Finite dimensional spaces
    RTk = FiniteElement('RT', mesh.ufl_cell(), deg+1)
    Pk = FiniteElement('DG', mesh.ufl_cell(), deg)
    Pkp1 = FiniteElement('CG', mesh.ufl_cell(), deg+1)

    Vh = FunctionSpace(mesh, MixedElement([RTk, Pk, Pkp1]))
    nn.append(Vh.dim())

    # test and trial functions
    du = TrialFunction(Vh)
    u = Function(Vh)
    sig, p, q = split(u)
    tau, v, w = TestFunctions(Vh)

    # exact solutions
    p_ex = Expression(str2exp(p_str), degree=6, domain=mesh)
    q_ex = Expression(str2exp(q_str), degree=5, domain=mesh)
    sig_ex = -grad(p_ex) - G(r2, p_ex, q_ex)
    f2_ex = div_rad(r1, r2, D*sig_ex)
    f3_ex = dot(grad(q_ex), r2vec) + r2**2*p_ex

    # boundary conditions
    sigD = project(sig_ex, Vh.sub(0).collapse())
    bcS = DirichletBC(Vh.sub(0), sigD, bdry, left)
    beta = sqrt(dot(r2vec, r2vec))

    # Weak forms
    lhs = dot(sig + G(r2, p, q), tau)*weight*dx \
            - p*div_rad(r1, r2, tau)*weight*dx \
            - v*div_rad(r1, r2, D*sig)*weight*dx \
            + (-div_rad(r1, r2, r2vec)*q + r2**2*p)*w*weight*dx \
            - dot(r2vec, grad(w))*q*weight*dx \
            + dot(r2vec, n)*q*w*weight*ds(left) \
            + dot(r2vec, n)*q*w*weight*ds(rest) \
            + stab/(deg+1)**3.5*avg(hK)**2*beta*dot(jump(grad(q)), n('+'))*dot(jump(grad(w)), n('+'))*weight*dS \
            + stab2*(hK*neg(dot(r2vec, n))**2 + 8./hK)*q*w*ds

    rhs = -p_ex*dot(tau, n)*weight*ds(right) \
            - p_ex*dot(tau, n)*weight*ds(rest) \
            - f2_ex*v*weight*dx \
            + f3_ex*w*weight*dx \
            - dot(r2vec, n)*q_ex*w*weight*ds(right) \
            + stab2*(hK*neg(dot(r2vec, n))**2 + 8./hK)*q_ex*w*ds

    FF = lhs - rhs

    Tang = derivative(FF, u, du)
    problem = NonlinearVariationalProblem(FF, u, J=Tang, bcs=bcS)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver'] = 'snes'
    solver.parameters['snes_solver']['linear_solver'] = 'mumps'
    solver.parameters['snes_solver']['absolute_tolerance'] = 1e-7
    solver.parameters['snes_solver']['relative_tolerance'] = 1e-7
    solver.solve()
    sig_h, p_h, q_h = u.split()

    # Error computations (can be omitted if not needed for timing/memory)
    E_q_H1 = assemble((q_ex-q_h)**2*weight*dx + dot(r2vec, grad(q_ex)-grad(q_h))**2*weight*dx)
    E_p_L2 = assemble((p_ex-p_h)**2*weight*dx)
    E_s_Hdiv = assemble((sig_ex-sig_h)**2*weight*dx + (div_rad(r1, r2, sig_ex-sig_h))**2*weight*dx)

    # Optionally, return errors if you want to log them
    return E_q_H1, E_p_L2, E_s_Hdiv

if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    for nk in range(nkmax):
        # Try to collect garbage to minimize memory drift
        import gc
        gc.collect()
        mem_before = process.memory_info().rss / 1024**2  # in MiB
        t0 = time.time()
        E_q_H1, E_p_L2, E_s_Hdiv = run_one_mesh(nk)
        t1 = time.time()
        mem_after = process.memory_info().rss / 1024**2  # in MiB
        print(f"nk={nk}: Mem before={mem_before:.2f} MiB, Mem after={mem_after:.2f} MiB, ΔMem={mem_after-mem_before:.2f} MiB, Time={t1-t0:.2f} s")
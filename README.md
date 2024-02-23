# What: Closest Molecule
Finite element methods for a nonlinear advecion-diffusion problem in radial/spherical coordinates. The implementation relies on the [FEniCS](https://fenicsproject.org) finite element libraries (tested with v.2019.1). Visualisation is done with [Paraview](https://paraview.org). For the mesh generation and manipulation we use the [FreeFem++](https://freefem.org) finite element library and the library [GMSH](https://gmsh.info)

# Problem definition 
Insert here brief description of the PDE 
...

Initial condition would look like this 
<img width="835" alt="Screenshot 2024-02-23 at 17 14 18" src="https://github.com/ruizbaier/closestMolecule/assets/29896148/02a93131-cc3e-49bf-a420-93f0c49a957d">


# Mesh generation

We could generate a geometry and triangular mesh in GMSH, save it into msh format (2, ascii) and then convert it to FEniCS format xml with dolfin-convert. However the curved boundary ( at $r\_1 = F(r\_2)$ ) is difficult to do (in GMSH it seems that one only has available Bezier, splines, etc).

So we generate the geometry and the triangular mesh using FreeFem++. Check [this file](https://github.com/ruizbaier/closestMolecule/meshes/meshWithExpReal.edp). It is run with 

```
> FreeFem++ meshWithExpReal.edp
```

It seems that saving directly to msh format does not work well. Therefore this script saves it automatically to mesh format. The problem is then that the physical entities are lost. So we use another script, [see here](https://github.com/ruizbaier/closestMolecule/meshes/changingEleToPhys.py). This generates the correct .msh mesh. We run it with 

```
> python3 changingEleToPhys.py
```

Then we convert the produced .msh file to something readable in FEniCS (e.g. xml format) with 

```
> dolfin-convert xxx.msh xxx.xml
```

This generates also a file containing the boundary tag information
<img width="835" alt="Screenshot 2024-02-23 at 17 16 13" src="https://github.com/ruizbaier/closestMolecule/assets/29896148/410d0432-b913-4053-900c-501f909a650e">

# Finite element discretisation

We simply use $\mathbb{P}\_1$ elements (piecewise linear and overall continuous) to approximate the solution. Essential (Dirichlet) boundary conditions are imposed in the trial space

$$V_h :=  \\{w\_h \in H\^1(\Omega): w_h|\_K \in \mathbb{P}\_1(K)\ \forall K\in \mathcal{T}_h, \quad w_h|\_{\Gamma^{\mathrm{top}}} = u_0, \ w_h|\_{\Gamma^{\mathrm{bot}}} = 0,\ w_h|\_{\Gamma^{\mathrm{right}}} = 0  \\}, $$

and the test space $V_h^0$ is the homogeneous counterpart of $V_h$:

$$V^0_h : = \\{ w\_h \in H\^1(\Omega): w_h|\_K \in \mathbb{P}\_1(K)\ \forall K\in \mathcal{T}_h, \quad w_h|\_{\Gamma^{\mathrm{top}}} = 0, \ w_h|\_{\Gamma^{\mathrm{bot}}} = 0,\ w_h|\_{\Gamma^{\mathrm{right}}} = 0\\}.$$


We use backward Euler's method for the time discretisation. The fully discrete form reads: find $u_h\in V\_h$ such that 

$$\int\_{\Omega} \frac{u\_h -u\_h\^n}{\Delta t} v_h + \int\_{\Omega} 
\begin{pmatrix} D_2 & 0 \\
0 & D_1 \end{pmatrix} 
\nabla u_h \cdot \nabla v_h - 2\int\_{\Omega}(\frac{D\_2}{r\_2}\partial\_{r\_2}u_h +\frac{D\_1}{r\_1}\partial\_{r\_1}u_h)v_h + \int_{\Omega} G(u_h)\begin{pmatrix} 1\\
0 \end{pmatrix} \cdot \nabla v_h - \int_{\Omega}\frac{2}{r_2} G(u_h) v_h, \qquad \forall v_h \in V^0_h;$$

where $G(u_h):= \frac{4\pi r_2^2u_h^2}{\int_{r_2}^\infty 4\pi r'^2u_h(r_1,r',t)\,dr'}\approx 4\pi r_2^2u_h^2$.

Check [this code](https://github.com/ruizbaier/closestMolecule/SphericalAdvectionDiffusionReaction_computingSols.py). It is run with 

```
> python3 SphericalAdvectionDiffusionReaction_computingSols.py
```

# Verification of convergence

These tests use the method of manufactured solutions to check the rate of convergence of the schemes (how the error decays with the meshsize). Check [this code](https://github.com/ruizbaier/closestMolecule/convergence/SphericalAdvectionDiffusionReaction_convergence.py). It gives as output a table with the error history: 

```
=================================================
  DoF      h    e_1(u)   r_1(u)   e_0(u)  r_0(u)    
=================================================
     9  0.7071 1.81e+00  0.000  1.74e-01  0.000 
    25  0.3536 9.23e-01  0.968  3.52e-02  2.305 
    81  0.1768 4.45e-01  1.050  7.51e-03  2.228 
   289  0.0884 2.20e-01  1.021  1.75e-03  2.102 
  1089  0.0442 1.09e-01  1.007  4.29e-04  2.030 
  4225  0.0221 5.46e-02  1.001  1.09e-04  1.971 
 16641  0.0110 2.73e-02  0.998  2.95e-05  1.891 
=================================================
```
which confirms the expected order of convergence ($h$ in the $H\^1(\Omega)$-norm and order $h^2$ in the $L\^2(\Omega)$-norm).

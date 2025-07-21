# What: Closest Molecule
Finite element methods for a nonlinear advection-diffusion problem in radial/spherical coordinates. The implementation relies on the [FEniCS](https://fenicsproject.org) finite element libraries (tested with v.2019.1). Visualisation is done with [Paraview](https://paraview.org). For the mesh generation and manipulation we use the [FreeFem++](https://freefem.org) finite element library and the library [GMSH](https://gmsh.info)

# Problem definition 
We study a system of diffusing point particles in which any triplet of particles reacts and is removed from the system when the relative proximity of the constituent particles satisfies a predefined condition. Such proximity-based reaction conditions are well understood for bimolecular systems, but generalising them to trimolecular reactions, those involving three particles, is complicated because it requires understanding the distribution of the closest diffusing particle to a point in the vicinity of a spatially dependent absorbing boundary condition.

Here we consider a domain $\Omega$ of volume $V$, where $V$ is finite but very large, that contains the three distinct chemical species, $A$, $B$ and $C$, whose molecules are all initially well mixed (distributed uniformly at random) within $\Omega$. The system contains a single molecule of $A$ and $B$, and $N_C = cV$ molecules of $C$ where $c$ is a well-mixed concentration of $C$ molecules. We seek to understand the evolution of the state, where each state consists of the molecules of $A$ and $B$ and a particular $C$ molecule, associated with the molecule of $C$ that is closest to the centre of diffusion of $A$ and $B$. The steady-state joint probability density $P(r_1,r_2)$ to find this particular state evolves according to the coupled system of PDEs

$$\mathbf{s}(r_1,r_2) + \left(\nabla P(r_1,r_2) + \frac{r_2^2 P(r_1,r_2)^2}{Q(r_1,r_2)}\hat{\mathbf{r}}_2\right) = 0,\quad
\nabla \cdot (D\mathbf{s}(r_1,r_2)) = 0,\quad \text{and} \quad
\frac{\partial Q(r_1,r_2)}{\partial r_2} + r_2^2P(r_1,r_2) = 0.$$

Here: 
- $r_1$ and $r_2$ denote the radial distance between the molecules of $A$ and $B$ and the radial distance from the centre of diffusion of these molecules to the molecule of $C$, respectively, and the associated unit vectors are

$$\hat{\boldsymbol{r}}\_1 = \begin{pmatrix} 0\\
1 \end{pmatrix} \quad \text{and} \quad \hat{\boldsymbol{r}}\_2 = \begin{pmatrix} 1\\
0 \end{pmatrix};$$

- $\mathbf{s}(r_1,r_2)$ is the flux of $P$; 
- D is a matrix of the diffusion coefficients such that

$$D = \begin{pmatrix} D_2 & 0 \\
0 & D_1 \end{pmatrix};$$

- $\nabla P(r_1,r_2)$ denotes the radial gradient of $P$,
  
$$\nabla P(r_1,r_2) = \frac{\partial P(r_1,r_2)}{\partial r_1}\hat{\mathbf{r}}_1 + \frac{\partial P(r_1,r_2)}{\partial r_2}\hat{\mathbf{r}}_2;$$

- $\nabla \cdot \mathbf{s}(r_1,r_2)$ denotes the radial divergence of $\mathbf{s}(r_1,r_2)$,
  
$$  \nabla \cdot \mathbf{s}(r_1,r_2) = \frac{1}{r_1^2}\frac{\partial}{\partial  r_1}\left(r_1^2\mathbf{s}(r_1,r_2) \cdot \hat{\mathbf{r}}_1\right) + \frac{1}{r_2^2}\frac{\partial}{\partial  r_2}\left(r_2^2\mathbf{s}(r_1,r_2) \cdot \hat{\mathbf{r}}_2\right);$$

- we have introduced the density moment
  
$$Q(r_1,r_2) = \int^\infty_{r_2} P(r_1, r_2')r_2'^2 \mathrm{d}r_2',$$

to avoid explicit calculation of the integral.

We divide the boundary ($\partial \Omega$) of the domain into four segments: $\Gamma_{\mathrm{left}}$, $\Gamma_{\mathrm{bot}}$, $\Gamma_{\mathrm{right}}$ and $\Gamma_{\mathrm{top}}$, defined by the regions $r_2 = 0$, $r_1 = f(r_2)$, $r_2 = r_2^{\text{max}}$ and $r_1 = r_1^{\text{max}}$, respectively. The boundary conditions applied on these segments are:

$$\mathbf{s}(r_1,r_2) \cdot \hat{\mathbf{r_2}} = 0 \quad \text{on } \Gamma_{\mathrm{left}},$$

$$\quad P(r_1,r_2) = 0 \quad \text{on } \Gamma_{\mathrm{bot}},$$ 

$$\quad P(r_1,r_2) = \frac{c}{V}\text{exp}\left(\frac{-4\pi c r_2^3}{3}\right)\left(1 - \frac{f(r_2)}{r_1}\right) \quad \text{on } \Gamma_{\mathrm{top}},$$

$$Q(r_1,r_2) = \frac{1}{4\pi V}\text{exp}\left(\frac{-4\pi c r_2^3}{3}\right) \quad \text{on } \Gamma_{\mathrm{right}} \quad \text{and}$$

$$ P(r_1,r_2) = \frac{c}{V}\text{exp}\left(\frac{-4\pi c r_2^3}{3}\right) \quad \text{on } \Gamma_{\mathrm{right}}.$$

# Mesh generation

We could generate a geometry and triangular mesh in GMSH, save it into msh format (2, ascii) and then convert it to FEniCS format xml with dolfin-convert. However the curved boundary ( at $r\_1 = f(r\_2)$ ) is difficult to do (in GMSH it seems that one only has available Bezier, splines, etc).

So we generate the geometry and the triangular mesh using FreeFem++. Check [this file](https://github.com/ruizbaier/closestMolecule/blob/main/meshes/meshWithExpReal.edp). It is run with 

```
> FreeFem++ meshWithExpReal.edp
```

It seems that saving directly to msh format does not work well. Therefore this script saves it automatically to mesh format. The problem is then that the physical entities are lost. So we use another script, [see here](https://github.com/ruizbaier/closestMolecule/blob/main/meshes/changingEleToPhys.py). This generates the correct .msh mesh. We run it with 

```
> python3 changingEleToPhys.py
```

Then we convert the produced .msh file to something readable in FEniCS (e.g. xml format) with 

```
> dolfin-convert xxx.msh xxx.xml
```

This also generates a file containing the boundary tag information.
<img width="835" alt="Screenshot 2024-02-23 at 17 16 13" src="https://github.com/ruizbaier/closestMolecule/assets/29896148/410d0432-b913-4053-900c-501f909a650e">

The mesh construction described above is handled automatically for each iteration of the boundary [see here](https://github.com/ruizbaier/closestMolecule/blob/main/correct_boundary_finite_element.py). This automation relies on [this file](https://github.com/ruizbaier/closestMolecule/blob/main/meshes/automated_exp_meshing.edp)
# Finite element discretisation

We simply use $\mathbb{P}\_1$ elements (piecewise linear and overall continuous) to approximate the solution. Essential (Dirichlet) boundary conditions are imposed in the trial space

$$V_h :=  \\{w\_h \in H\^1(\Omega): w_h|\_K \in \mathbb{P}\_1(K)\ \forall K\in \mathcal{T}_h, \quad w_h|\_{\Gamma^{\mathrm{top}}} = u_0, \ w_h|\_{\Gamma^{\mathrm{bot}}} = 0,\ w_h|\_{\Gamma^{\mathrm{right}}} = 0  \\}, $$

and the test space $V_h^0$ is the homogeneous counterpart of $V_h$:

$$V^0_h : = \\{ w\_h \in H\^1(\Omega): w_h|\_K \in \mathbb{P}\_1(K)\ \forall K\in \mathcal{T}_h, \quad w_h|\_{\Gamma^{\mathrm{top}}} = 0, \ w_h|\_{\Gamma^{\mathrm{bot}}} = 0,\ w_h|\_{\Gamma^{\mathrm{right}}} = 0\\}.$$


We use backward Euler's method for the time discretisation. The fully discrete form reads: find $u_h\in V\_h$ such that 

$$\int\_{\Omega} \frac{u\_h -u\_h\^n}{\Delta t} v_h + \int\_{\Omega} \mathbf{D} \nabla u_h \cdot \nabla v_h - 2\int\_{\Omega}(\frac{D\_2}{r\_2}\partial\_{r\_2}u_h +\frac{D\_1}{r\_1}\partial\_{r\_1}u_h)v_h + \int_{\Omega} G(u_h)\hat{\boldsymbol{r}}\_2 \cdot \nabla v_h - \int_{\Omega}\frac{2}{r_2} G(u_h) v_h \quad \forall v_h \in V^0_h;$$

where we are taking also the approximation $G(u_h) \approx D_2 4\pi r_2^2u_h^2$ (implying a constant probability density). 

Check [this code](https://github.com/ruizbaier/closestMolecule/blob/main/SphericalAdvectionDiffusionReaction_computingSols.py). It is run with 

```
> python3 SphericalAdvectionDiffusionReaction_computingSols.py
```

# Verification of convergence

These tests use the method of manufactured solutions to check the rate of convergence of the schemes (how the error decays with the meshsize). Check [this code](https://github.com/ruizbaier/closestMolecule/blob/main/convergence/SphericalAdvectionDiffusionReaction_convergence.py). It gives as output a table with the error history: 

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

# closestMolecule
finite element methods for a nonlinear advecion-diffusion problem in radial/spherical coordinates

# Problem definition 
Initial condition would look like this 
<img width="835" alt="Screenshot 2024-02-23 at 17 14 18" src="https://github.com/ruizbaier/closestMolecule/assets/29896148/02a93131-cc3e-49bf-a420-93f0c49a957d">


# Mesh generation

We could generate a geometry and triangular mesh in GMSH, save it into msh format (2, ascii) and then convert it to FEniCS format xml with dolfin-convert. However the curved boundary (at r1 = F(r2)) is difficult to do (in GMSH it seems that one only has available Bezier, splines, etc).

So we generate the geometry and the triangular mesh using FreeFem++. The file is meshWithExpReal.edp. It seems that saving directly to msh format does not work well. Therefore this script saves it to mesh format.
The problem is then that the physical entities are lost. So we use another script, changingEleToPhys.py. This generates the correct .msh mesh. Then we convert it to FEniCS format xml with dolfin-convert

<img width="835" alt="Screenshot 2024-02-23 at 17 16 13" src="https://github.com/ruizbaier/closestMolecule/assets/29896148/410d0432-b913-4053-900c-501f909a650e">

# Finite element discretisation

We simply use P1 elements (piecewise linear and overall continuous) to approximate the solution. We use backward Euler's method for the time discretisation. 

# Verification of convergence

These tests use the method of manufactured solutions to check the rate of convergence of the schemes (how the error decays with the meshsize). Check SphericalAdvectionDiffusionReaction_convergence.py. 

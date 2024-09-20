from fenics import *

"""
Creates a simple rectangular boundary mesh for testing.
"""


# mesh construction.
TOL = DOLFIN_EPS
R1_MIN = 0.1
R1_MAX = 0.5
R2_MIN = 0
R2_MAX = 2.0
MESH_RES_R1 = 100
MESH_RES_R2 = 200

class boundary_L(SubDomain):
    """
    The inner boundaries of the mesh
    """
    def inside(self, x, on_boundary):
        return near(x[0], R2_MIN, TOL) and on_boundary

class boundary_B(SubDomain):
    """
    The inner boundaries of the mesh
    """
    def inside(self, x, on_boundary):
        return near(x[1], R1_MIN, TOL) and on_boundary

class boundary_R(SubDomain):
    """
    The inner boundaries of the mesh
    """
    def inside(self, x, on_boundary):
        return near(x[0], R2_MAX, TOL) and on_boundary

class boundary_T(SubDomain):
    """
    The inner boundaries of the mesh
    """
    def inside(self, x, on_boundary):
        return near(x[1], R1_MAX, TOL) and on_boundary


mesh = RectangleMesh(Point(R2_MIN,R1_MIN),Point(R2_MAX,R1_MAX),MESH_RES_R2,MESH_RES_R1)
plot(mesh)
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
# All facets labeled 0 now.
sub_domains.set_all(0)

# Mark the boundaries.
right = 21
top = 22
left = 23
bottom = 24

boundary_R().mark(sub_domains, right)
boundary_T().mark(sub_domains, top)
boundary_L().mark(sub_domains, left)
boundary_B().mark(sub_domains, bottom)

# Save the mesh and the subdomain data to files for use elsewhere.
File("rect_boundary.xml") << mesh
File("rect_boundary_facet_region.xml") << sub_domains
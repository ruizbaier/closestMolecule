from fenics import *

"""
Creates a simple rectangular boundary mesh for testing.
"""


# mesh construction.
TOL = DOLFIN_EPS


class BoundaryL(SubDomain):
    """
    The bottom boundary of the mesh
    """

    def __init__(self, R2_min):
        super(BoundaryL, self).__init__()
        self.R2_min = R2_min

    def inside(self, x, on_boundary):
        return near(x[0], self.R2_min, TOL) and on_boundary


class BoundaryB(SubDomain):
    """
    The bottom boundary of the mesh
    """

    def __init__(self, R1_min):
        super(BoundaryB, self).__init__()
        self.R1_min = R1_min

    def inside(self, x, on_boundary):
        return near(x[1], self.R1_min, TOL) and on_boundary


class BoundaryR(SubDomain):
    """
    The right boundary of the mesh
    """
    def __init__(self, R2_max):
        super(BoundaryR, self).__init__()
        self.R2_max = R2_max

    def inside(self, x, on_boundary):
        return near(x[0], self.R2_max, TOL) and on_boundary


class BoundaryT(SubDomain):
    """
    The top boundary of the mesh
    """
    def __init__(self, R1_max):
        super(BoundaryT, self).__init__()
        self.R1_max = R1_max

    def inside(self, x, on_boundary):
        return near(x[1],self.R1_max, TOL) and on_boundary


def create_mesh(r1_min, r1_max, r2_min, r2_max, mesh_base_res, filename='rect_boundary'):
    mesh_res_r1 = int(mesh_base_res * (r1_max - r1_min))
    mesh_res_r2 = int(mesh_base_res * (r2_max - r2_min))
    mesh = RectangleMesh(Point(r2_min, r1_min), Point(r2_max, r1_max), mesh_res_r2, mesh_res_r1)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # All facets labeled 0 now.
    sub_domains.set_all(0)
    # Mark the boundaries.
    right = 21
    top = 22
    left = 23
    bottom = 24
    BoundaryR(r2_max).mark(sub_domains, right)
    BoundaryT(r1_max).mark(sub_domains, top)
    BoundaryL(r2_min).mark(sub_domains, left)
    BoundaryB(r1_min).mark(sub_domains, bottom)
    # Save the mesh and the subdomain data to files for use elsewhere.
    File("meshes/" + filename + ".xml") << mesh
    File("meshes/" + filename + "_facet_region.xml") << sub_domains


if __name__ == '__main__':
    R1_min = 0.05
    R1_max = 0.5
    R2_min = 0
    R2_max = 1.0
    Mesh_base_res = 1000
    create_mesh(R1_min, R1_max, R2_min, R2_max, Mesh_base_res)


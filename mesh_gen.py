import numpy as np
from dolfin import *
from fenics import *
from mshr import Polygon, generate_mesh
import matplotlib.pyplot as plt

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
        return x[1] <= self.R1_min and on_boundary


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

def boundary_func(sigma, gamma, r2, *a0):
    boundary = sigma*np.exp(-4 * np.pi * gamma * np.power(r2, 3) / 3) - a0[0]
    try:
        boundary[boundary < 0] = 0
    except TypeError:
        if boundary < 0:
            boundary = 0
    return boundary


def construct_vertices(sigma, gamma, r1_max, r2_max, num_bottom_points, *a0):
    domain_vertices = []
    # Add top left corner
    domain_vertices.append(Point(0, r1_max))
    # Add bottom boundary vertices
    r2_values = np.linspace(0, r2_max, num_bottom_points)
    r1_values = boundary_func(sigma, gamma, r2_values, a0[0])
    points = np.column_stack((r2_values, r1_values))
    '''
    for point in points:
        print(point)
    input()
    '''
    for i in range(len(r2_values)):
        domain_vertices.append(Point(r2_values[i], r1_values[i]))
    # Add top right corner
    domain_vertices.append(Point(r2_max, r1_max))
    # Close the polygon
    domain_vertices.append(Point(0, r1_max))
    return domain_vertices

def construct_mesh(sigma, gamma, r1_max, r2_max, bottom_points, mesh_filename, *a0):
    domain_vertices = construct_vertices(sigma, gamma, r1_max, r2_max, bottom_points, a0[0])
    domain = Polygon(domain_vertices)
    mesh = generate_mesh(domain, 20*r1_max)
    #plot(mesh)
    #plt.show()

    for j in range(1):
        refine_cell = MeshFunction("bool", mesh, mesh.topology().dim())
        for c in cells(mesh):
            vertices_x = sorted(c.get_vertex_coordinates()[::2])
            vertices_y = sorted(c.get_vertex_coordinates()[1::2])
            # Three vertices for each cell
            for i in range(3):
                if vertices_y[i] <= 0.5:
                    refine_cell[c] = True
                else:
                    refine_cell[c] = False
        mesh = refine(mesh, refine_cell)
        #plot(mesh)
        #plt.show()
    # Print the mesh stats
    print(f'Number of cells {mesh.num_cells()} max size {mesh.hmax()} min size {mesh.hmin()}')
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
    BoundaryL(0).mark(sub_domains, left)
    BoundaryB(sigma).mark(sub_domains, bottom)
    # Save the mesh and the subdomain data to files for use elsewhere.
    File("meshes/" + mesh_filename + ".xml") << mesh
    File("meshes/" + mesh_filename + "_facet_region.xml") << sub_domains

if __name__ == '__main__':
    r1_vals = np.array([5])
    for r1_val in r1_vals:
        sigma = 0.1
        r1_max = r1_val
        r2_max = 5
        gamma = 1
        construct_mesh(sigma, gamma, r1_max, r2_max, 1000, f'exp_mesh_sigma{sigma}_r1max{r1_max}_r2max{r2_max}')
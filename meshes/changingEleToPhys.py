import gmsh
gmsh.initialize()
gmsh.merge("flat_boundary2.mesh") # or gmsh.open

for i in range(1,4):
    E = gmsh.model.getEntities(i)
    for ei in E:
        gmsh.model.addPhysicalGroup(i, [ei[1]], ei[1])



#        gmsh.save("meshWithExpReal2.msh")

gmsh.option.setNumber("Mesh.MshFileVersion",2.2) 
gmsh.write("flat_boundary2.msh")

# doing stuff

gmsh.finalize()  

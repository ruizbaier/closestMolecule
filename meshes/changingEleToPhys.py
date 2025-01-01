import gmsh
folder = "sigma_test/"
filenames = ["exp_boundary_sigma0.5on8pi_gamma1_r1max5_r2max5.mesh", "exp_boundary_sigma0.75on8pi_gamma1_r1max5_r2max5.mesh",
"exp_boundary_sigma1on8pi_gamma1_r1max5_r2max5.mesh", "exp_boundary_sigma1.25on8pi_gamma1_r1max5_r2max5.mesh",
"exp_boundary_sigma1.5on8pi_gamma1_r1max5_r2max5.mesh", "exp_boundary_sigma1.75on8pi_gamma1_r1max5_r2max5.mesh",
"exp_boundary_sigma2on8pi_gamma1_r1max5_r2max5.mesh", "exp_boundary_sigma2.25on8pi_gamma1_r1max5_r2max5.mesh",
"exp_boundary_sigma2.5on8pi_gamma1_r1max5_r2max5.mesh", "exp_boundary_sigma2.75on8pi_gamma1_r1max5_r2max5.mesh",
"exp_boundary_sigma3on8pi_gamma1_r1max5_r2max5.mesh"]
for file in filenames:
    gmsh.initialize()
    gmsh.merge(folder + file) # or gmsh.open

    for i in range(1,4):
        E = gmsh.model.getEntities(i)
        for ei in E:
            gmsh.model.addPhysicalGroup(i, [ei[1]], ei[1])

    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
    gmsh.write(folder + file[:-4]+"msh")

    # doing stuff

    gmsh.finalize()

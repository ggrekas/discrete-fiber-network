#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

main script to run simulations
------------------------------------------------------------------------------------
3 files output: 
i) .txt with the simulation's parameters, title, etc.
ii) two HDF5 files for mesh & u
------------------------------------------------------------------------------------

"""

# ======================================== #
# Libraries
# ======================================== #

from SOLVER import *
import time
from datetime import datetime
import os
import pickle

# ======================================== #
# Parameters
# ======================================== #

date = datetime.now().strftime("%m-%d-%Y")
path0 ='results'
if not os.path.exists(path0 +'/'):
    os.makedirs(path0)

# --- Define Domain and cell radii:
c_x, c_y, rho = 0., 0., 20     # center and radius of main (domain) circle

# --- Uncomment for one or two cells simulation
c1_x, c1_y, i_rho1 = 0., 0., 2      # center and radius of the first small circle   ONE CELL SIMULATION

c_d = 12   # centers distance  TWO CELL SIMULATION
# c1_x, c1_y, i_rho1 = c_d/2.0, 0., 2      # center and radius of the first small circle
# c2_x, c2_y, i_rho2 = -c_d/2.0, 0., 2     # center and radius of the second small circle

# --- Boundary conditions: contraction => u0 = -a*r, r cell radius
u0 = None # Domain Boundary : Traction free -> None , else 0.0 for fixed domain boundary

# --- Tractions on cell or domain boundary : always NONE
Tn = None
Tn_r = None
Tn_l = None

# --- Problem formulation:
resolution = 60 # 10, 20, 30, ...
CG_order = 1 # always 1 (This is the degree for the finite element function space, i.e. piecewise continuous polynomial of first degree)
n = 0 # reduce connectivity: choose number of nodes to be removed -> n=0 zero edges are removed | n=1 , one edge is removed from each vertex.
k = 1.0 	# same constant for each facet 

# - Define  cell(s) contraction: (un)comment accordingly
a = np.arange(0.05, 0.5, 0.05) # 10 iterations for the specified contractions
# a = np.array([0.5]) # single contraction , eg 0.5 accounts for 50 % cell contraction

# - Choose constitutive law:
law = "L5_L3" # provide S(lamda) as a string : 'L3_L', 'L5_L3', 'L7_L5', 'L7_1', 'L5_1', 'L3_1', 'L_1'
print("Law: {}, n= {}".format(law, n))

# - Choose optimization algorithm:
# method = 'BFGS' # quasi Newton
method = 'conj' # nonlnr conj

# --- --- --- 
ttl = "Single cell simulation" # title of the particular run or set of runs
print(ttl)
dtnow = datetime.now().strftime("%m-%d-%Y_%I-%M-%S_%p")
fname = '{}_{}'.format(law,dtnow) # prefix of the output files

# - Right parameter set on txt file:
with open('{}/{}.txt'.format(path0,date), 'a') as file:
    
    file.write('\n\n# ======================================== #')
    file.write('\nTitle: {}'.format(ttl))
    file.write('\nConstitutive. Law: {}'.format(law))
    file.write('\nReduce connect by n = {}'.format(n))
    # file.write('\nNumber of cells: {}'.format(str(1)))   # 1 or 2 or n cells 
    file.write('\nk: {}'.format(str(k)))
    file.write('\nRadii: {}, {}'.format(rho,i_rho1))
    file.write('\nResolution: {}'.format(str(resolution)))
    file.write('\nDomain Boundary Condition: {}'.format(str(u0)))
    # file.write('\nCell distance: {}'.format(str(c_d)))
    # file.write('\nContraction: {}'.format(str(a)))
    file.write('\n- - - -\n')
    file.write('Number of simulations: {}'.format(str(len(a))))
    file.write('\nFilename prefix:')
    file.write('\n {}'.format(fname))

# ======================================== #
# Formulation and optimization:
# ======================================== #

# refine = True; print("Refine: TRUEEE") # added for random mesh simulations

for i in range(len(a)): # iteration over contraction array 
    mesh_name = '{}_{}_mesh.hdf5'.format(fname,str(i))
    u_name = '{}_{}_u.hdf5'.format(fname,str(i))

    u0_r = (-1)*a[i]*i_rho1 # single cell simulations
    # u0_l = (-1)*a[i]*i_rho2 # uncomment this one for two-cells contraction
    
    # --- Domain formulation:
    domain = CircularDomain(rho, c_x, c_y,u0,Tn)
    right_circle = CircularDomain(i_rho1, c1_x, c1_y,u0_r,Tn_r, bound_res=50) # single cell simulations
    # left_circle = CircularDomain(i_rho2, c2_x, c2_y, u0_l,Tn_l, bound_res=50) # uncomment this one for two-cells contraction

    # --- Remove cells from mesh:
    # circles = [right_circle,left_circle] # uncomment this one for two-cells contraction
    circles = [right_circle] # single cell simulations
    [domain.remove_subdomain(cell) for cell in circles]

    # --- Problem formulation:
    problem_obj = Problem(domain,resolution,CG_order,law,k) # PROBLEM call
    solver  = Solver(problem_obj,n) # typical SOLVER call
    # solver  = Solver(problem_obj,n,refine) # for mesh randomization SOLVER call 
    
    start_time = time.time()
    solver.minimization(method)
    fin_time = time.time()
    dt = fin_time - start_time
    seconds = int(dt)
    print('Time to create-solve problem : {} minutes'.format(dt/60.0))
    
    u = solver.get_u()
    solver.save_mesh_to_HDF5('{}/{}'.format(path0,mesh_name)) # save HDF5 mesh
    solver.save_to_HDF5(u,'{}/{}'.format(path0,u_name)) # save HDF5 u

    if n !=0: # for reduce connectivity save a pickle with the indices of the edges that have been removed
        facets_old = solver.get_FacetsDict_init()
        C1 = solver.get_connectivity(facets_old)
        print("\nConnectivity of initial mesh: {}".format(C1))
        name_file = "{}/{}_{}_RemovedEdges.pkl".format(path0,fname,str(i))
        facets_new = solver.get_FacetsReduced()
        C2 = solver.get_connectivity(facets_new)
        print("Connectivity of refined mesh: {}".format(C2))
        edges_out = solver.get_edges_removed()#; print (edges_out)
        with open(name_file, 'wb') as fp:
            pickle.dump(edges_out, fp)
    
    with open('{}/{}_{}.txt'.format(path0,fname,str(i)), 'w') as file:
        file.write('\nTitle: {}'.format(ttl))
        file.write('\nReduce connect by n = {}'.format(n))
        file.write('\nContraction: {} -> '.format(a[i]))
        file.write(' {}'.format(mesh_name))
        file.write(', {}'.format(u_name))
        file.write('\nTime: {} minutes'.format(dt/60.0))
        file.write('\n--------------------------------------------\n')
        file.write('\n--------------------------------------------\n')
        file.write('\nConstitutive Law: {}'.format(law))
        # file.write('\nFinal Connectivity = {}'.format(C2)) # only for reduce connectivity simulations
        file.write('\nNumber of cells: {}'.format(str(len(circles))))  # 1, 2 etc for number of cells
        file.write('\nRadii: {}, {}'.format(rho,i_rho1))
        file.write('\nResolution: {}'.format(str(resolution)))
        file.write('\nk: {}'.format(str(k)))
        file.write('\nBoundary Condition: {}'.format(str(u0)))
        # file.write('\nCell distance: {}'.format(str(c_d))) # for two cells

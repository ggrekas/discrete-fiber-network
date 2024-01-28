#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@date: 20/11/2020
version: python 3

---------------------------------------



"""

from PROBLEM import*
from scipy import optimize
from matplotlib import pyplot as plt


class Solver:
    def __init__(self, problem,n, refine = None):
        
        self._problem = problem
        self._edges_to_cut = n
        self._randomize_mesh = refine

        if self._randomize_mesh:
            self._problem.randomize_mesh_nodes(self._randomize_mesh)
        else:
            self._problem.extract_mesh()
        
        #self._rest_lngth = self._problem.extract_rest_lengths_old()
        self._rest_lngth_ar = self._problem.extract_rest_lengths()
        self._mesh = self._problem.get_mesh()
        self._coordinates = self._problem.get_coordinates()
        self._EdgesDict = self._problem.get_FacetsDict()
        self._kapa = self._problem.set_constant()
        self._cell_areas = self._problem.extract_rest_areas()
        self._refined_fibers_idx = self._problem.get_refined_fibers_idx()
        
        self._problem.set_VectorFunctionSpace()
        self._V = self._problem.get_V()
        self._idx_VertexToDof = self._problem.get_VertexToDof()
        self._idx_DofToVertex = self._problem.get_DofToVertex()
        
        self._u = self._problem.initialization()
        self._detF = None
    
        self._u_array_mapped = self._u.vector().get_local()[self._idx_VertexToDof]
        self._free_nodes_idx, self._load_nodes_idx = self._problem.free_and_load_nodes(self._u_array_mapped)
        
        # -----
        # Reduce connectivity:
        if self._edges_to_cut != 0:
            self._EdgesDict_reduced, self._edges_removed = self._problem.reduce_connectivity(self._mesh, self._edges_to_cut)
            self._kapa[self._edges_removed] = 0
            self._numbr_of_paired_vrtx, self._paired_vertx_to_node, self._edges_to_vert_idx = self._problem.extract_connections_to_free_nodes(self._EdgesDict_reduced)
        else:
            self._numbr_of_paired_vrtx, self._paired_vertx_to_node, self._edges_to_vert_idx = self._problem.extract_connections_to_free_nodes(self._EdgesDict)
        
        self._free_nodes_extend = self._problem.get_free_nodes_extend_ar()
        
        self._number_of_cells_for_each_vtx, self._cell_areas_idx_for_each_vtx, self._idxnode_a ,self._idxnode_b = self._problem.extract_triangle_arrays()
        self._free_nodes_extend_per_cell = self._problem.get_free_nodes_extend_bycell()
        self._init_cross = self._problem.get_init_cross_product()
        # print(self._init_cross[0:200])

        # -----
        # self._Nodes_dict = self._problem.extract_FreeNodesDict()  # Initial implementation (slow)
        # self._Nodes_PerCell_dict = self._problem.extract_NodesPerCell() # Initial implementation (slow)
        # -----

        self._domain = self._problem.get_domain()
        self._circles = self._problem.get_circles()
        

    def minimization(self,method):
        
        u_array_mapped = self._u_array_mapped
        coordinates = self._coordinates
        
        free_nodes_idx = self._free_nodes_idx
        constant = self._kapa
        nodes_prime_init = u_array_mapped + coordinates
        #------
        if self._edges_to_cut != 0:
            facets_dict = self._EdgesDict_reduced
        else:
            facets_dict = self._EdgesDict
        rest_lngth = self._rest_lngth_ar
        # Nodes_dict = self._Nodes_dict
        # Nodes_of_cell = self._Nodes_PerCell_dict
        areas = self._cell_areas 
        #-------
        if self._problem.check_for_traction_bc() == False:
            Fext_array = []
            load_nodes_idx = []
        else:
            Fext =  self._problem.extract_traction_array()
            n = Fext.shape[0]

            load_nodes_idx = self._load_nodes_idx
            Fext_array = np.split(Fext, n/2)
        
        # -----
        # args = (u_array_mapped,coordinates, facets_dict, rest_lngth, Nodes_dict, Nodes_of_cell, nodes_prime_init,\
        #     free_nodes_idx, Fext_array, load_nodes_idx,constant,areas)
        #-------
        args = (u_array_mapped,coordinates,nodes_prime_init,free_nodes_idx, Fext_array, load_nodes_idx,constant)
        
        if method == 'BFGS':
            u_array, fopt, d = optimize.fmin_l_bfgs_b(func= self._problem.Energy,
                x0= u_array_mapped[free_nodes_idx], fprime = self._problem.Gradient,
                args = args,iprint = 0) 
    
        else:
            u_array, fopt, f_calls, g_calls, warnflag = optimize.fmin_cg(self._problem.Energy, 
                u_array_mapped[free_nodes_idx],  fprime=self._problem.Gradient,
                args = args,full_output=True,disp = True)
            print (fopt)

        u_array_mapped[free_nodes_idx] = u_array
        # print u_array
        self._u.vector().set_local(u_array_mapped[self._idx_DofToVertex])
        
        return
    
    
    def get_connectivity(self, facets):
        
        Nodes_old = self._problem.edges_through_node_idx(facets)
        conn = []
        for node in Nodes_old:
            conn.append(len(node))

        C = np.mean(conn)
        return C

   
    # ==============================================================================
    #   Plot Functions:
    
    def mesh_plot(self):

        mesh = self._mesh

        try:
            plot(mesh, interactive = True)
        except:
            plot(mesh)
            plt.show()

    def displacement_plot(self):

        u = self.get_u()

        try:
            plot(u,mode='displacement', interactive = True)
        except:
            plot(u,mode='displacement')
            plt.show()


    def detF_plot(self, J = None):
        mesh = self._mesh
        u = self.get_u()
        if J == None:
            J = self.get_detF()

        new_mesh = Mesh(mesh)
        V0 = FunctionSpace(new_mesh, 'DG', 0)
        detF_proj = project(J, V0)
        ALE.move(new_mesh, u)
        
        try:
            plot(detF_proj,interactive=True)
        except:
            plot(detF_proj)
            plt.show()


    # ==============================================================================
    #   Save results:

    def save_to_pvd(self,obj,name):
        file = File(name)
        file << obj
    
    def save_mesh_to_HDF5(self,name):
        mesh = self._mesh
        try:
            mesh_file = HDF5File(MPI.comm_world, name, 'w')
            mesh_file.write(mesh, 'mesh')
        except:
            mesh_file = HDF5File(mpi_comm_world(), name, 'w')
            mesh_file.write(mesh, 'mesh')

    def save_to_HDF5(self,obj,name):
        mesh = self._mesh
        try:
            f = HDF5File(mesh.mpi_comm(), name, "w")
            # f = HDF5File(MPI.comm_world, name, "w")
            f.write(obj, "initial")
        except:
            f = HDF5File(mesh.mpi_comm(), name, "w")
            f.write(obj, "initial")

    
    def read_mesh_HDF5(self,name):
        mesh = Mesh()
        hdf5 = HDF5File(mpi_comm_world(), name, 'r')
        hdf5.read(mesh, 'mesh', False)
        return mesh

    def read_u_HDF5(self,name):
        
        V = self._V
        u = Function(V)
        mesh = self._mesh
        
        f = HDF5File(mesh.mpi_comm(), name, "r")
        f.read(u, "initial")
        return u
    
    def read_J_HDF5(self,u,name):

        mesh = self._mesh
        new_mesh = Mesh(mesh)
      
        V0 = FunctionSpace(new_mesh, 'DG', 0)
    
        J = Function(V0)
        f = HDF5File(new_mesh.mpi_comm(), name, "r")
        f.read(J, "/initial")
        detF_proj = project(J, V0)
        ALE.move(new_mesh, u)
        return detF_proj
    # ==============================================================================

    def get_u(self):
        return self._u

    def get_FreeNodes(self):
        return self._free_nodes_idx

    def get_TractionNodes(self):
        return self._load_nodes_idx
        
    def get_detF(self):

        u = self.get_u()
        d = u.geometric_dimension()
        mesh = self._mesh

        I = Identity(d)  # Identity tensor
        F = variable(I + grad(u))  # Deformation gradient
        J = variable(det(F))
        V0 = FunctionSpace(mesh, 'DG', 0)
        self._detF = project(J, V0)
        return self._detF

    def get_NodesDict(self):
        return self._Nodes_dict
    def get_NodesPerCell(self):
        return self._Nodes_PerCell_dict
    def get_areas(self):
    	return self._cell_areas

    def get_FacetsDict_init(self):
        return self._EdgesDict
    def get_FacetsReduced(self):
        return self._EdgesDict_reduced
    def get_k(self):
        return self._kapa
    def get_edges_removed(self):
        return self._edges_removed
    def get_mesh(self):
        return self._mesh

    def return_refined_fibers_idx(self):
        return self._refined_fibers_idx
# ==============================================================================
# ==============================================================================

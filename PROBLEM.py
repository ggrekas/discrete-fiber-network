#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@date: 20/11/2020
version: python 3

Models added
Comments added
In reduce connectivity: keep cell geometry
-----------------------------------------


"""

from DOMAIN import*
from matplotlib import pyplot as plt

# ===================================================================

class Problem:
    def __init__(self, domain,resolution, CG_order,Law,constant = None):
        
        self._domain = domain
        self._circles = self._domain.get_subDomainsList()
        self._resolution = resolution
        self._CG_order = CG_order
        self._kapa = constant

        self._mesh = None
        self._coordinates = None
        self._EdgesDict = None
        self._ElementsDict = None
        #self._restlngth = None
        self._restlngth_ar = None
        self._cell_areas = None
        self._no_of_nodes = None
        self._e_coor_indx = None
        self._refined_fibers_idx = None # only in mesh randomization


        self._u = None
        self._V = None
        self._v = None
        self._idx_VertexToDof = None
        self._idx_DofToVertex = None
        self._detF = None
        self._J = None
        self._V0 = None

        
        self._free_nodes_idx = None
        self._fixed_nodes_idx = None
        self._load_nodes_idx = None
        self._free_vertex = None

        # --- 
        self._Nodes_dict_new = None
        self._Nodes_PerCell_dict = None
        # ---
        self._subDomainIndex = []
        self._ConstitutiveLawDict = {"LinearSprings":self.linear_springs, "Grekas":self.grekas, "L3_L":self.WL3_L,"L5_L3":self.WL5_L3,"L7_L5":self.WL7_L5, "L_1":self.WL_1, "L3_1":self.WL3_1, "L5_1":self.WL5_1, "L7_1":self.WL7_1}
        self._GradientDict = {"LinearSprings":self.linear_springs_gradient, "Grekas":self.grekas_gradient, "L3_L":self.L3_L, "L5_L3":self.L5_L3, "L7_L5":self.L7_L5, "L_1":self.L_1, "L3_1":self.L3_1, "L5_1":self.L5_1, "L7_1":self.L7_1}
        self._energy = Law  

        # !!! additions for vectorizing the Gradient()
        self._numbr_of_paired_vrtx  = None
        self._paired_vertx_to_node = None
        self._edges_to_vert_idx = None
        self._free_nodes_extend = None

        self._number_of_cells_for_each_vtx = None
        self._cell_areas_idx_for_each_vtx = None
        self._idxnode_a = None
        self._idxnode_b = None
        self._free_nodes_extend_per_cell = None
        self._init_cross = None

    def extract_mesh(self):
        """
        Input: NO
        Output: initialize all related mesh attributes
        """
        domain, circles = self._domain, self._circles

        domain.create_mesh(self._resolution)

        self._mesh = domain.get_mesh()
        self._coordinates = np.hstack(self._mesh.coordinates())
        self._EdgesDict = dict((edge.index(), edge.entities(0)) for edge in edges(self._mesh))
       
        self._mesh.init()
        self._e_coor_indx =  np.zeros( 2*self._mesh.num_edges() ).astype(np.int) 
        for edge in edges(self._mesh):
            i, j = edge.entities(0)
            self._e_coor_indx[2*edge.index()] = i
            self._e_coor_indx[2*edge.index()+1] = j

        self._ElementsDict = dict((cell.index(), cell.entities(0)) for cell in cells(self._mesh))
        self._iter = 0
        self._no_of_nodes = int(self._coordinates.shape[0]/2)
        return 

    def randomize_mesh_nodes(self, m):  # !!!! TEST FUNCTION: should be cleaned up
        np.random.seed(1) # to produce exact same less connected mesh for multiple iterations

        domain, circles = self._domain, self._circles

        domain.create_mesh(self._resolution)
        mesh = domain.get_mesh() # Do not store as self._mesh !!
        coordinates = np.hstack(mesh.coordinates())

        EdgesDict = dict((edge.index(), edge.entities(0)) for edge in edges(mesh))
        mesh.init()
        e_coor_indx =  np.zeros( 2*mesh.num_edges() ).astype(np.int)
        for edge in edges(mesh):
            i, j = edge.entities(0)
            e_coor_indx[2*edge.index()] = i
            e_coor_indx[2*edge.index()+1] = j
        
        ElementsDict = dict((cell.index(), cell.entities(0)) for cell in cells(mesh))
        self._iter = 0
        self._no_of_nodes = int(coordinates.shape[0]/2)

        # --- Apply BC : initialize u & V:
        V = VectorFunctionSpace(mesh, 'CG', self._CG_order)
        # self._V0 = FunctionSpace(mesh0, 'DG', 0)
        idx_VertexToDof = vertex_to_dof_map(V) # mapping
        idx_DofToVertex = dof_to_vertex_map(V) # inverse mapping
    
        u = Function(V)
        bcs = domain.create_bc(V)
        [bc.apply(u.vector()) for bc in bcs]
        u_array_mapped = u.vector().get_local()[idx_VertexToDof]

        # ---
        # d = self._u.geometric_dimension()
        # I = Identity(d)  # Identity tensor
        # F2 = variable(I + grad(self._u))  # Deformation gradient
        # J2 = variable(det(F2))
        # V0 = FunctionSpace(mesh, 'DG', 0)
        # detF = project(J2, V0)
        # Jo_array = detF.vector().get_local();print(Jo_array[0:100])

        # ----

        # new_mesh0 = Mesh(mesh)
        # V02 = FunctionSpace(new_mesh0, 'DG', 0)
        # detF_proj = project(detF, V02)
        # ALE.move(new_mesh0, self._u)
        # plot(1.0/detF_proj)
        # plt.show()
        # V0 = FunctionSpace(mesh, 'DG', 0)
        # detF = project(self._J, self._V0)
        # Jo_array = detF.vector().get_local();print(Jo_array)

        # --- Rest Lengths and pre-tension:
        coor = mesh.coordinates()
        dif_coor = coor[e_coor_indx[::2]] - coor[e_coor_indx[1::2]]
        restlngth_ar =  np.linalg.norm(  dif_coor , axis=1)#; print("initial rest lengths : {}".format(restlngth_ar[0:100]))
        # --- Compute each fiber-vector:
       
        fibers_bndr = self.save_boundary_fibers(e_coor_indx, V)#;print(fibers_bndr)
        # The second node comes first in substracting !!!!
        diff_coor_x = coordinates[2*e_coor_indx[1::2]] - coordinates[2*e_coor_indx[::2]] 
        diff_coor_y = coordinates[2*e_coor_indx[1::2]+1] - coordinates[2*e_coor_indx[::2]+1]

        new_length = m*np.mean(restlngth_ar) #- np.std(restlngth_ar)
        print("Initial mean fiber length: {}".format(np.mean(restlngth_ar)))
        print("Initial std : {}".format(np.std(restlngth_ar)))
        print("Assigned length: {}".format(new_length))
        idx_all = np.where(restlngth_ar > np.mean(restlngth_ar )- np.std(restlngth_ar))[0] # fibers_to_extend or shrink
        # idx_all = np.where(restlngth_ar > np.mean(restlngth_ar )- 0*np.std(restlngth_ar))[0] # fibers_to_extend or shrink 
        print(len(idx_all))
        print(len(EdgesDict))
        idx = np.asarray(sorted(list(set(idx_all) - set(fibers_bndr))))#;print(idx) # exclude boundary fibers
        # idx: final fiber indices for those to be refined --> save !!
        self._refined_fibers_idx = idx
        # print(idx_all)
        x_coor_prime = (new_length/restlngth_ar[idx])* diff_coor_x[idx]#;print(x_coor_prime) # of refined fiber
        y_coor_prime = (new_length/restlngth_ar[idx])* diff_coor_y[idx]#;print(y_coor_prime) # of refined fiber

        x_node_prime = coordinates[2*e_coor_indx[::2]][idx] + x_coor_prime
        y_node_prime = coordinates[2*e_coor_indx[::2]+1][idx] + y_coor_prime

        du_x = x_node_prime - coordinates[2*e_coor_indx[1::2]][idx]
        du_y = y_node_prime - coordinates[2*e_coor_indx[1::2] +1][idx]

        u_array_mapped[2*e_coor_indx[2*idx + 1]] = du_x
        u_array_mapped[2*e_coor_indx[2*idx + 1] + 1] = du_y
        # -----
        # idx2_all = np.where((restlngth_ar < new_length) & (restlngth_ar > new_length -np.std(restlngth_ar)))[0]
        # # idx2 = np.where(restlngth_ar < new_length -np.std(restlngth_ar))[0] # fibers_to_extend
        # idx2 = np.asarray(sorted(list(set(idx2_all) - set(fibers_bndr))))
        # # idx2 = np.asarray(sorted(list(set(idx22) - set(idx))));print(idx2)

        # bb = new_length -2*np.std(restlngth_ar)
        # x_coor_prime = (bb/restlngth_ar[idx2])* diff_coor_x[idx2]#;print(x_coor_prime) # of refined fiber
        # y_coor_prime = (bb/restlngth_ar[idx2])* diff_coor_y[idx2]#;print(y_coor_prime) # of refined fiber

        # x_node_prime = coordinates[2*self._e_coor_indx[::2]][idx2] + x_coor_prime
        # y_node_prime = coordinates[2*self._e_coor_indx[::2]+1][idx2] + y_coor_prime

        # du_x = x_node_prime - coordinates[2*self._e_coor_indx[1::2]][idx2]
        # du_y = y_node_prime - coordinates[2*self._e_coor_indx[1::2] +1][idx2]

        # u_array_mapped[2* self._e_coor_indx[2*idx2 + 1]] = du_x
        # u_array_mapped[2* self._e_coor_indx[2*idx2 + 1] + 1] = du_y


        # --- Assign in global variables:
        DD = self.save_boundary_indices(V)
        # print(DD)
        
        QQ = np.asarray(list(DD.values()))#;print(QQ.shape)
        QQ2 = np.concatenate(QQ, axis =0)#;print(QQ2)
        
        u_ar_copy = np.copy(u_array_mapped)
        # u_ar_copy[DD['right']] = np.zeros(len(DD['right']))
        u_ar_copy[QQ2] = np.zeros(len(QQ2))

        u.vector().set_local(u_ar_copy[idx_DofToVertex]) # update u array after changes
        
        new_mesh = Mesh(mesh)
        ALE.move(new_mesh, u)
        self._mesh = new_mesh
        # plot(new_mesh)
        # plt.show()

        # self._u.vector().set_local(u_array_mapped[idx_DofToVertex])
        self._coordinates = np.hstack(self._mesh.coordinates())
        self._EdgesDict = dict((edge.index(), edge.entities(0)) for edge in edges(self._mesh))
       
        self._mesh.init()
        self._e_coor_indx =  np.zeros( 2*self._mesh.num_edges() ).astype(np.int) 
        
        for edge in edges(self._mesh):
            i, j = edge.entities(0)
            self._e_coor_indx[2*edge.index()] = i
            self._e_coor_indx[2*edge.index()+1] = j

        self._ElementsDict = dict((cell.index(), cell.entities(0)) for cell in cells(self._mesh))
        
        return 

    def initialization(self):
        """
        Apply boundary condition and initialize Deformation gradient
        """
        V = self._V
        self._u = Function(V)
        self.apply_bc(self._u)
        d = self._u.geometric_dimension()
        
        I = Identity(d)  # Identity tensor
        F = variable(I + grad(self._u))  # Deformation gradient
        self._J = variable(det(F))
        
        return self._u

    def extract_rest_lengths(self,mesh0 = None):
        """
        Extract rest lengths of all fibers (edges) in a single array
        """
        if not mesh0:
            coor, e_coor_indx  = self._mesh.coordinates(), self._e_coor_indx
        else:
            coor , e_coor_indx = mesh0.coordinates() , self._e_coor_indx
        
        dif_coor = coor[e_coor_indx[::2]] - coor[e_coor_indx[1::2]]
        self._restlngth_ar =  np.linalg.norm(  dif_coor , axis=1)
        # print("Rest lengths: {}".format(self._restlngth_ar[0:100]))
        return self._restlngth_ar

    def extract_rest_areas(self):
        """
        Extract rest area of all elements (triangles) in a single array
        """
        mesh = self._mesh
        no_of_elements = len(self._ElementsDict)
        self._cell_areas = np.zeros(no_of_elements)

        for i, cell in enumerate(cells(mesh)):
            self._cell_areas[i] = cell.volume()
        return self._cell_areas
    
    def set_VectorFunctionSpace(self, mesh0 = None):
        """
        Set VectorFunctionSpace & related indices-mapping lists
        """

        if not mesh0:
            mesh0 = self._mesh

        self._V = VectorFunctionSpace(mesh0, 'CG', self._CG_order)
        self._V0 = FunctionSpace(mesh0, 'DG', 0)
        self._idx_VertexToDof = vertex_to_dof_map(self._V) # mapping
        self._idx_DofToVertex = dof_to_vertex_map(self._V) # inverse mapping
        return

    def set_constant(self):
        """
        Set stiffness constant for all fibers:
        i. If single constant is given, assign the same to all
        ii. If not, assign random constant to each 
        """
        size = len(self._EdgesDict)
        if self._kapa is None:
            self._kapa = np.random.sample(size)  # never tested. Need for literature search for accepted stiffness values
        else:
            self._kapa = np.repeat(self._kapa,size) 
        return self._kapa

    def extract_boundary_index(self, domain_obj, Vo = None):
        """
        Input: domain_obj -> the boundary facet/circle
        Output: the indices of input boundary
        
        """
        
        if Vo is None:
            Vo = self._V
            
        idx_VertexToDof = vertex_to_dof_map(Vo)
        
        wall = domain_obj.get_bnd_function()

        u_temp = Function(Vo)

        # --- Define TEMPORARY Boundary conditions:
        temp = Expression(("1.0", "1.0"), degree=1)
        bc_temp = DirichletBC(Vo, temp, wall)     # (1,1) for the given boundary

        # --- Apply the TEMPORARY Boundary conditions:
        bc_temp.apply(u_temp.vector())

        # --- Mapping from vertices to degrees of freedom & extract free nodes:

        u_temp_mapped = u_temp.vector().get_local()[idx_VertexToDof]

        # --- EXTRACT the indices of the nodes of the boundary: 
        boundary_nodes_idx = np.nonzero(u_temp_mapped)[0]

        return boundary_nodes_idx

    def return_idx_of_vertices(self,list_of_double_indx):
        """
        Input: domain_obj -> the boundary facet/circle
        Output: the indices of input boundary
        """
        keep_idx = list_of_double_indx[::2]
        nodes_idx = [int(x/2) for x in keep_idx]
        # !!! if keep_idx == array: no need for comrehension; only because it's list !!! TO CHECK
        return nodes_idx
    
    
    def free_and_load_nodes(self, u_array_mapped):
        """
        Input: u as array [after mapping]
        Output: indices of free nodes & load nodes 
        """
        circles = self._circles
        domain = self._domain

        nodes_with_bc = []
        
        for cell in circles:
            if cell.has_bc() == True:
                cell_idx = self.extract_boundary_index(cell)
                nodes_with_bc.append(cell_idx)
            else:
                if cell.has_traction_bc() == True:
                    cell_idx = self.extract_boundary_index(cell)
                    self._subDomainIndex.append(cell_idx)

        if domain.has_bc() == True:
            print ("| Domain is fixed | ")
            domain_nodes = self.extract_boundary_index(domain)
            nodes_with_bc.append(domain_nodes)
        else:
            if domain.has_traction_bc() == True:
                print ("| External domain has traction field | ")
                domain_idx = self.extract_boundary_index(domain)
                self._subDomainIndex.append(domain_idx)

        
        self._fixed_nodes_idx = [idx for i in nodes_with_bc for idx in i]
        self._load_nodes_idx = [ix for j in self._subDomainIndex for ix in j]
        
        u_mapped_idx_list = range(len(u_array_mapped))
        
        self._free_nodes_idx = [i for i in u_mapped_idx_list if i not in self._fixed_nodes_idx]

        self._free_vertex = self.return_idx_of_vertices(self._free_nodes_idx) # new object added: used in later functions
        
        return self._free_nodes_idx, self._load_nodes_idx 
    
    # =============================================================
    def extract_NodesPerCell(self):
        """
        Store info per element: 
        - indices of connected nodes (by pairs)
        - reference area
        Output: stored in Dictionary  
        """
        mesh, coordinates = self._mesh, self._coordinates
        areas = self._cell_areas
        nodes = range(self._no_of_nodes)
        mesh_cells = mesh.cells()
        free_nodes_idx = self._free_nodes_idx
        
        Nodes_dict_II = {}

        for i in nodes:
            couples = []
            volume = []
            for j in range(mesh_cells.shape[0]):  # lista me listes apo ta nodes tou kathe cell
                # print "j= {}".format(j)
                if i in mesh_cells[j]:
                    volume.append(areas[j])
                    pair_nodes_idx = np.where(mesh_cells[j] != i)[0]
                    pair_nodes = mesh_cells[j][pair_nodes_idx]#; print pair_nodes
                    couples.append(pair_nodes)
            Nodes_dict_II.update({i: [couples,volume]})
    

        self._Nodes_PerCell_dict = {int(k): Nodes_dict_II[k] for k in self._free_vertex}

        return self._Nodes_PerCell_dict

    def extract_FreeNodesDict(self):
        
        coordinates, facets = self._coordinates,self._EdgesDict
        free_nodes_idx = self._free_nodes_idx
        nodes = range(self._no_of_nodes)
        
        Nodes_dict = {}
        for i in nodes:
            vertices = []
            lengths = []
            edges = []
            for key in facets.keys():
                if i in facets[key]:
                    edges.append(key)
                    
                    pair_node_idx = np.where(facets[key] != i)[0][0]
                    pair_node = facets[key][pair_node_idx]
                    vertices.append(pair_node)
                    # ---
                    pi = np.array([coordinates[2*i],coordinates[(2*i)+1]])
                    pj = np.array([coordinates[2*pair_node],coordinates[(2*pair_node)+1]])

                    Lo = np.sqrt(np.sum((pi - pj)**2) )
                    lengths.append(float(Lo))
            Nodes_dict.update({i: [vertices, lengths, edges]})

        
        self._Nodes_dict_new = {int(k): Nodes_dict[k] for k in self._free_vertex}

        return self._Nodes_dict_new
    
    
    # ==============================================
    def extract_connections_to_free_nodes(self, edge_dict):
        '''
        For FREE nodes: extract in arrays
            i. vertices conncted to them
            ii.edges to these vertices

        instead of the !!! Nodes_dict_new !!!
        '''
        #coordinates, edge_dict = self._coordinates,self._EdgesDict
        coordinates = self._coordinates
        idx_of_nodes = self._free_vertex
        
        # how_many_vertices are paired with each free node:
        numbr_of_paired_vrtx = []
        # index of each paired node, for ALL free nodes:
        paired_vertx_to_node = []
        # index of EDGE to each paired node, for ALL free nodes:
        edges_to_vert_idx = []

        for i in idx_of_nodes:
            vertices = 0
            for key in edge_dict.keys():
                if i in edge_dict[key]:
                    edges_to_vert_idx.append(key)
                    
                    pair_node_idx = np.where(edge_dict[key] != i)[0][0]
                    pair_node = edge_dict[key][pair_node_idx]
                    paired_vertx_to_node.append(pair_node)
                    vertices += 1
            numbr_of_paired_vrtx.append(vertices)

        self._numbr_of_paired_vrtx = np.asarray(numbr_of_paired_vrtx)
        self._paired_vertx_to_node = np.asarray(paired_vertx_to_node)
        self._edges_to_vert_idx = np.asarray(edges_to_vert_idx)
        
        # Extend the free_vertex array, so that each node is repeated as many times as the number of its connections:
        self._free_nodes_extend  = np.repeat(self._free_vertex,np.asarray(numbr_of_paired_vrtx))

        return self._numbr_of_paired_vrtx, self._paired_vertx_to_node, self._edges_to_vert_idx
    
    
    def extract_triangle_arrays(self):
        '''
        For FREE nodes: extract in arrays
            i. paired vertices conncted to them per triangle !
            ii.edges to these vertices
            iii. indices of the nodes that create the "a" vector
            iv. indices of the nodes that create the "b" vector
            v. reference cross product of all elements: 
                np.cross(a,b) for each a,b formated from each free node

        instead of the !!! self._Nodes_PerCell_dict !!!
        '''
        coordinates, cell_rest_areas = self._coordinates, self._cell_areas
        mesh_cells = self._mesh.cells()
        idx_of_nodes = self._free_vertex
        
        number_of_cells_for_each_vtx = []
        cell_areas_idx_for_each_vtx = []
        idxnode_a = []
        idxnode_b = []

        for node in idx_of_nodes:
            count_cells = 0
            for j in range(mesh_cells.shape[0]):  # lista me listes apo ta nodes tou kathe cell
                if node in mesh_cells[j]:
                    count_cells +=1
                    cell_areas_idx_for_each_vtx.append(j)
                    pair_nodes_idx = np.where(mesh_cells[j] != node)[0]
                    pair_nodes = mesh_cells[j][pair_nodes_idx]#; print (pair_nodes)
                    idxnode_a.append(pair_nodes[0])
                    idxnode_b.append(pair_nodes[1])
            number_of_cells_for_each_vtx.append(count_cells)

        self._number_of_cells_for_each_vtx = np.asarray(number_of_cells_for_each_vtx)
        self._cell_areas_idx_for_each_vtx = np.asarray(cell_areas_idx_for_each_vtx)
        self._idxnode_a = np.asarray(idxnode_a)
        self._idxnode_b = np.asarray(idxnode_b)

        
        # Extend the free_vertex array, so that each node is repeated as many times as the number of its connections:
        self._free_nodes_extend_per_cell  = np.repeat(self._free_vertex,np.asarray(number_of_cells_for_each_vtx))

        # ===== Reference cross product for each free node:

        xa_x0 = coordinates[2*self._idxnode_a] - coordinates[2*self._free_nodes_extend_per_cell]
        ya_y0 = coordinates[2*self._idxnode_a + 1] - coordinates[2*self._free_nodes_extend_per_cell + 1]

        xb_x0 = coordinates[2*self._idxnode_b] - coordinates[2*self._free_nodes_extend_per_cell]
        yb_y0 = coordinates[2*self._idxnode_b + 1] - coordinates[2*self._free_nodes_extend_per_cell + 1]

        aref = np.array([xa_x0, ya_y0])
        bref = np.array([xb_x0, yb_y0])

        self._init_cross = np.cross(aref,bref, axis=0)
        
        return self._number_of_cells_for_each_vtx, self._cell_areas_idx_for_each_vtx, self._idxnode_a, self._idxnode_b
    
    # ===================================================================
    # CONNECTIVITY:

    def edges_through_node_idx(self,facets):

        nodes = range(self._no_of_nodes)
        Nodes_list = []
        
        for i in nodes:
            edges = []
            for key in facets.keys():
                if i in facets[key]:
                    
                    pair_node_idx = np.where(facets[key] != i)[0][0]
                    pair_node = facets[key][pair_node_idx]
                    edges.append(key)
            Nodes_list.append(edges)

        return Nodes_list

    def reduce_connectivity(self, mesh,n):
        np.random.seed(1) # to produce exact same less connected mesh for multiple iterations

        facets_dict_old = dict((facet.index(), facet.entities(0)) for facet in facets(mesh))
        Nodes_list_old = self.edges_through_node_idx(facets_dict_old)

        all_nodes_idx = range(self._no_of_nodes)
        at_least_one_edge = []
        to_remove = []

        # edges on cell(s) boundary , to keep !! :
        edges_on_cell_bdry = list(self.save_boundary_fibers())

        for node in all_nodes_idx:
            lista = []
            for key in facets_dict_old.keys():
                if node in facets_dict_old[key]:
                    lista.append(key)
            edge = np.random.choice(lista,1)
            if edge[0] not in at_least_one_edge:
                at_least_one_edge.append(edge[0])

        for key in  Nodes_list_old:
            edges_choose = np.random.choice(key,n,replace=False)
            edges_choose_idx = [np.where(key == i) for i in edges_choose]
            edges_choose_idx = [item[0] for sublist in edges_choose_idx for item in sublist]

            edges = np.take(key,edges_choose_idx)
            to_remove.append(edges)

        # indices of edges to be removed: k->0
        edges_tobe_removed = [item for sublist in to_remove for item in sublist]
        keep_edges_conct = at_least_one_edge + edges_on_cell_bdry
        # final_edges_to_be_cut = sorted(list(set(edges_tobe_removed) - set(at_least_one_edge))) # extract fibers from cell bndr as well
        final_edges_to_be_cut = sorted(list(set(edges_tobe_removed) - set(keep_edges_conct))) # keep cell bndry fibers in

        facets_new = copy.deepcopy(facets_dict_old)
        for key in list(facets_new):
            # print(key)
            if key in final_edges_to_be_cut:
                del facets_new[key]

        return facets_new, final_edges_to_be_cut

    # ===================================================================
    
    def apply_bc(self, u0):

        V = self._V
        domain = self._domain

        bcs = domain.create_bc(V)
        [bc.apply(u0.vector()) for bc in bcs]

    def check_for_traction_bc(self):

        c = self._circles
        domain = self._domain
        
        if all(cell.has_traction_bc() == False for cell in c) == True:
            if domain.has_traction_bc() == False:
                print ("| NO traction field applied to any domain !!")
                return False
        print ("| Traction field to some domain(s)")

    
    def extract_traction_array(self):
        
        V = self._V
        idx_VertexToDof = self._idx_VertexToDof

        circles = self._circles
        domain = self._domain
        subDomainIndex = self._subDomainIndex

        Traction = [] 
        counter = 0
        
        for cell in circles:
            if cell.has_traction_bc() == True:
                Tn = cell.traction_on_bnd()
                cell_idx = subDomainIndex[counter] # mapped index

                u_in = interpolate(Tn,V)
                F_cell = u_in.vector().get_local()[idx_VertexToDof][cell_idx]
                Traction.append(F_cell)
                counter += 1

        if domain.has_traction_bc() == True:
            Tn = domain.traction_on_bnd()
            dom_idx = subDomainIndex[-1] # mapped index

            u_in = interpolate(Tn,V)
            F_dom = u_in.vector().get_local()[idx_VertexToDof][dom_idx]
            Traction.append(F_dom)
       
        if len(Traction) > 1:
            Traction = np.concatenate(Traction)
        else:
            Traction = Traction[0]
        
        return Traction


    # ==============================================================================
    #   Deformation Gradient:

    def deformation_gradient(self,u):
        d = u.geometric_dimension()
        
        I = Identity(d)  # Identity tensor
        F = variable(I + grad(u))  # Deformation gradient
        J = variable(det(F))
        self._detF = project(J, self._V0)

        return self._detF


    # ===================================================================
    # Energy Functions:

    def linear_springs(self,*args):
        """
        Initial implementation
        Input: lo -> the rest length & lprime -> deformed length
        Output: The strain energy of specific facet
        """
        lo,lprime,k = args
        lamba = lprime/lo

        U = (1.0/2.0)*k*(lprime-lo)**2 #+ k*lo/lprime
        return U

    def grekas(self,*args):
        """
        Initial implementation fro buckling : L5_L3
        """
        lo,lprime,k = args
        
        lamba = lprime/lo
        W_lambda = k*lo*(lamba**6/6 - lamba**4/4 -1.0/6.0 +1.0/4.0 ) 
        return W_lambda
    #     return 0.0
    
    def WL3_L(self, *args):
        lo,lprime,k = args
        
        lamba = lprime/lo
        W_lambda = k*lo*(lamba**4/4 - lamba**2/2 -1.0/4.0 +1.0/2.0 ) 
        return W_lambda
        # return 0.0

    def WL5_L3(self, *args):
        lo,lprime,k = args
        
        lamba = lprime/lo
        W_lambda = k*lo*(lamba**6/6 - lamba**4/4 -1.0/6.0 +1.0/4.0 ) 
        return W_lambda
        # return 0.0

    def WL7_L5(self, *args):
        lo,lprime,k = args
        
        lamba = lprime/lo
        W_lambda = k*lo*(lamba**8/8 - lamba**6/6 -1.0/8.0 +1.0/6.0 ) 
        return W_lambda
        # return 0.0
    
    def WL_1(self, *args):
        lo,lprime,k = args
        
        lamba = lprime/lo
        W_lambda = k*lo*(lamba**2/2 - lamba -1.0/2.0 +1.0) 
        return W_lambda
        # return 0.0
    
    def WL3_1(self, *args):
        lo,lprime,k = args
        
        lamba = lprime/lo
        W_lambda = k*lo*(lamba**4/4 - lamba -1.0/4.0 +1.0) 
        return W_lambda
        # return 0.0
    
    def WL5_1(self, *args):
        lo,lprime,k = args
        
        lamba = lprime/lo
        W_lambda = k*lo*(lamba**6/6 - lamba -1.0/6.0 +1.0) 
        return W_lambda
        # return 0.0

    def WL7_1(self, *args):
        lo,lprime,k = args
        
        lamba = lprime/lo
        W_lambda = k*lo*(lamba**8/8 - lamba -1.0/8.0 +1.0) 
        return W_lambda
        # return 0.0
    
    # ===================================================================
    # Gradient of Energy functions:
    
    def linear_springs_gradient(self,*args):
        '''
        Initial implementation for linear springs
        '''
        Lp,Lo,Xs,Ys,lamba,k = args
        dx = (k*Xs*(Lp - Lo) / Lp ) #-k*(x1-x2)*Lo/Lp**3
        dy = (k*Ys*(Lp - Lo) / Lp ) #-k*(y1-y2)*Lo/Lp**3
        return dx,dy
        
    def grekas_gradient(self, *args):
        '''
        Initial implementation for buckling: L5_L3
        '''
        Lp,Lo,Xs,Ys,lamba,k = args
        
        dx = k*((lamba**5 - lamba**3)*Xs/Lp )
        dy = k*((lamba**5 - lamba**3)*Ys/Lp )
        # dx, dy = 0.0, 0.0        
        return dx,dy
    
    def L3_L(self, *args):
        Lp,Lo,Xs,Ys,lamba,k = args
        
        dx = k*((lamba**3 - lamba)*Xs/Lp )
        dy = k*((lamba**3 - lamba)*Ys/Lp )
        # dx, dy = 0.0, 0.0        
        return dx,dy

    def L5_L3(self, *args):
        Lp,Lo,Xs,Ys,lamba,k = args
        
        dx = k*((lamba**5 - lamba**3)*Xs/Lp )
        dy = k*((lamba**5 - lamba**3)*Ys/Lp )
        # dx, dy = 0.0, 0.0        
        return dx,dy
    
    def L7_L5(self, *args):
        Lp,Lo,Xs,Ys,lamba,k = args
        
        dx = k*((lamba**7 - lamba**5)*Xs/Lp )
        dy = k*((lamba**7 - lamba**5)*Ys/Lp )
        # dx, dy = 0.0, 0.0        
        return dx,dy

    def L_1(self, *args):
        Lp,Lo,Xs,Ys,lamba,k = args
        
        dx = k*((lamba - 1)*Xs/Lp )
        dy = k*((lamba - 1)*Ys/Lp )
        # dx, dy = 0.0, 0.0        
        return dx,dy
    
    def L3_1(self, *args):
        Lp,Lo,Xs,Ys,lamba,k = args
        
        dx = k*((lamba**3 - 1)*Xs/Lp )
        dy = k*((lamba**3 - 1)*Ys/Lp )
        # dx, dy = 0.0, 0.0        
        return dx,dy
    
    def L5_1(self, *args):
        Lp,Lo,Xs,Ys,lamba,k = args
        
        dx = k*((lamba**5 - 1)*Xs/Lp )
        dy = k*((lamba**5 - 1)*Ys/Lp )
        # dx, dy = 0.0, 0.0        
        return dx,dy
    
    def L7_1(self, *args):
        Lp,Lo,Xs,Ys,lamba,k = args
        
        dx = k*((lamba**7 - 1)*Xs/Lp )
        dy = k*((lamba**7 - 1)*Ys/Lp )
        # dx, dy = 0.0, 0.0        
        return dx,dy
    

    # ===================================================================
    # --- Penalty energy & gradient:

    def phi(self,Jo,areas):  # !!!! Fixed constants for the penalty -50 & -1/4 : all runs have been done with these
        
        Jo_array = Jo.vector().get_local()
        phiJo = np.sum(np.exp(-50*(Jo_array-1.0/4.0))*areas)
        return phiJo

    def dphi(self,Jo):
        dphiJ = (-50)*np.exp(-50*(Jo-1.0/4.0))
        return dphiJ
    

    # ===================================================================


    def Energy(self,x, *args):
        """
        Input: u displacement array, *args: tuple of system's fixed parameters including:
                mesh coordinates & a dictionary with items(): {Index of each facet: (Indices of the nodes it connects,rest length)}
        Output: The TOTAL energy of the mesh.
        ---------------------------------------------------------------------------------------

        Iteration over facets & calculation of deformed length & then, the energy for each facet, using StrainEnergy().
        Summmation over all energy values (of all facets) to gain the total energy for the mesh.

        """

        
        W = self._ConstitutiveLawDict[self._energy]

        u_array_mapped,coordinates,nodes_prime, free_nodes_idx, Fext_array,load_nodes_idx,constant = args

        # --- Position Vector (after displacement u):
        nodes_prime[free_nodes_idx] = coordinates[free_nodes_idx] + x
        u_array_mapped[free_nodes_idx] = x
      
        # --- Update u vector & compute J:
        self._u.vector().set_local(u_array_mapped[self._idx_DofToVertex])
        Ji = project(self._J, self._V0)
        phi_Ji = self.phi(Ji,self._cell_areas)

        # --- 
        total_energy = 0.0
        total_load_energy=0.0
     
        diff_coor_x = nodes_prime[2*self._e_coor_indx[::2 ]] -nodes_prime[2*self._e_coor_indx[1::2 ] ] 
        diff_coor_y = nodes_prime[2*self._e_coor_indx[::2]+1] -nodes_prime[2*self._e_coor_indx[1::2]+1 ] 
        defrm_lngth_ar =  np.sqrt(diff_coor_x**2 + diff_coor_y**2)
        
        rest_lngth_ar = self._restlngth_ar
        eargs = (rest_lngth_ar,defrm_lngth_ar,constant)
        energy_ar = W(*eargs)
        total_energy  = energy_ar.sum()

        if not Fext_array:
            total_load_energy = 0
        else:
            boundary_primes = nodes_prime[load_nodes_idx]
            boundary_primes_re = boundary_primes.reshape(int(len(boundary_primes)/2),2)
            
            # --- inner product(F, x) , x: node on boundary where F is applied 
            inner_products =  np.einsum('ij,ij->i', boundary_primes_re, Fext_array )
            
            # --- summation over all inner products:
            total_load_energy = np.sum(inner_products)
        
        
        U = total_energy - total_load_energy + phi_Ji
        
        # if self._iter%10 ==0:
        #     print("iter = " + repr(self._iter), "energy: "+repr(U))
        # self._iter +=1
        return U

    
    def Gradient(self, x, *args):
        """
        Input: u displacement array, *args: tuple of system's fixed parameters including:
                a. mesh coordinates 
                b. a dictionary with items(): {Index of each facet: (Indices of the nodes it connects,rest length)}
                c. an array with the rest legth of all favets.
                d. a dictionary with keys: the nodes of the mesh, and values the facets that pass through each node
                e. initial guess (as displacement) 
                f. a list with the indices of the FREE nodes
                
        Output: The gradient array for the given Energy() function
        ---------------------------------------------------------------------------------------

        Iteration over all mesh'es nodes. The code looks at each facet that comes through a particular node
        & computes the Gradient, for each direction, taking into account the index of the node of interest in each facet;
        this is necessary in order to decide the sign of the gradient at each node/direction.


        """
    
        dW = self._GradientDict[self._energy]
        u_array_mapped,coordinates,nodes_prime, free_nodes_idx, Fext_array,load_nodes_idx,constant = args

        # load_nodes = []
        # for i in range(0,len(load_nodes_idx),2): # !!!! out & asarray
        #     load_nodes.append(load_nodes_idx[i]/2)

        # --- Position Vector (after displacement u):
        nodes_prime[free_nodes_idx] = coordinates[free_nodes_idx] + x
        
        # --- --- --- --- Gradient over lamda:
        xs_dif = nodes_prime[2*self._free_nodes_extend] - nodes_prime[2*self._paired_vertx_to_node]
        ys_dif = nodes_prime[2*self._free_nodes_extend +1] - nodes_prime[2*self._paired_vertx_to_node +1]
        
        Lp =  np.sqrt(xs_dif**2 + ys_dif**2)
        Lo = self._restlngth_ar[self._edges_to_vert_idx]
        lamda = np.divide(Lp,Lo)

        k = constant[self._edges_to_vert_idx]
        gargs = (Lp,Lo,xs_dif,ys_dif,lamda,k)
        DX, DY = dW(*gargs)

        # sum the derivative for each node:
        numbr_paired_cum = np.cumsum(self._numbr_of_paired_vrtx, dtype = int)
        dX= np.split(DX, numbr_paired_cum)
        dY= np.split(DY, numbr_paired_cum)
        s_dx = np.array([np.sum(i) for i in dX if i.size])
        s_dy = np.array([np.sum(j) for j in dY if j.size])
        
        # --- --- --- --- Gradient over element - phi(J):
        
        area = self._cell_areas[self._cell_areas_idx_for_each_vtx]
        # new positions & coordinates for a,b : 
        fxa_x0 = nodes_prime[2*self._idxnode_a] - nodes_prime[2*self._free_nodes_extend_per_cell]
        fya_y0 = nodes_prime[2*self._idxnode_a + 1] - nodes_prime[2*self._free_nodes_extend_per_cell+ 1]
        
        fxb_x0 = nodes_prime[2*self._idxnode_b] - nodes_prime[2*self._free_nodes_extend_per_cell]
        fyb_y0 = nodes_prime[2*self._idxnode_b + 1] - nodes_prime[2*self._free_nodes_extend_per_cell + 1]
        
        aprm = np.array([fxa_x0, fya_y0])
        bprm = np.array([fxb_x0, fyb_y0])
        
        # Cross products & J:
        init_cross = self._init_cross
        def_cross = np.cross(aprm,bprm, axis=0)

        fya_fyb = nodes_prime[2*self._idxnode_a + 1] - nodes_prime[2*self._idxnode_b + 1]
        fxb_fxa = nodes_prime[2*self._idxnode_b] - nodes_prime[2*self._idxnode_a]
        
        J = def_cross/init_cross; #print (J)
        dJ = self.dphi(J)
                
        dx = dJ*fya_fyb*(1.0/init_cross)*area
        dy = dJ*fxb_fxa*(1.0/init_cross)*area

        numbr_paired_cum_2 = np.cumsum(self._number_of_cells_for_each_vtx, dtype = int)
        JdX= np.split(dx, numbr_paired_cum_2)
        JdY= np.split(dy, numbr_paired_cum_2)
        
        j_dx = np.array([np.sum(i) for i in JdX if i.size])
        j_dy = np.array([np.sum(j) for j in JdY if j.size])
        
        # --- --- --- --- STORE in dU: the gradient array
        dU = np.zeros(len(free_nodes_idx))
        dU[::2] = s_dx + j_dx
        dU[1::2] = s_dy + j_dy

        # --- --- --- --- Numerical Gradient: To test validity for own implementation 
            # gradF = np.asarray(dU)
            # args = (u_array_mapped,coordinates,nodes_prime, free_nodes_idx, Fext_array,load_nodes_idx,constant)
            # n = len(nodes_prime[free_nodes_idx])
            # h = 1e-7
            # numerDF = np.zeros(n)
            # for i in range(n):
            #     y = x.copy()
            #     y[i] += h
            #     numerDF[i] = (self.Energy(y,*args) - self.Energy(x.copy(),*args))/ h

            # print ('grad diff =', np.max(np.abs(gradF - numerDF)))

        return dU
    
    
    # --- ------------------------- old (slow) implementations for Energy() & Gradient():
    # def grekas(self,*args):
    #     lo,lprime,k = args
        
    #     lamba = lprime/lo
    #     W_lambda = k*lo*(lamba**6/6 - lamba**4/4 -1.0/6.0 +1.0/4.0 ) 
    #     return W_lambda
    #     # return 0.0

    # def grekas_gradient(self, *args):

    #     x1,y1,x2,y2, Lo,k = args
    #     Lp = np.sqrt( (x1-x2)**2 + (y1-y2)**2 ) # deformed length
    #     lamba = Lp/Lo
        
    #     dx = k*((lamba**5 - lamba**3)*(x1-x2)/Lp)
    #     dy = k*((lamba**5 - lamba**3)*(y1-y2)/Lp)
    #     # dx, dy = 0.0, 0.0
    #     dW = np.array([dx,dy])
    #     return dW

        
    # def Energy(self,x, *args):
    #     """
    #     Input: u displacement array, *args: tuple of system's fixed parameters including:
    #             mesh coordinates & a dictionary with items(): {Index of each facet: (Indices of the nodes it connects,rest length)}
    #     Output: The TOTAL energy of the mesh.
    #     ---------------------------------------------------------------------------------------

    #     Iteration over facets & calculation of deformed length & then, the energy for each facet, using StrainEnergy().
    #     Summmation over all energy values (of all facets) to gain the total energy for the mesh.

    #     """

        
    #     W = self._ConstitutiveLawDict[self._energy]

    #     u_array_mapped,coordinates,Facets_dict,rest_lngth,Nodes_dict, Nodes_of_cell,\
    #         nodes_prime, free_nodes_idx, Fext_array,load_nodes_idx,constant,areas = args

    #     # --- Position Vector (after displacement u):
    #     nodes_prime[free_nodes_idx] = coordinates[free_nodes_idx] + x
    #     u_array_mapped[free_nodes_idx] = x
      
    #     # --- Update u vector & compute J:
    #     self._u.vector().set_local(u_array_mapped[self._idx_DofToVertex])
    #     Ji = project(self._J, self._V0)
    #     phi_Ji = self.phi(Ji,areas)

    #     total_energy = 0.0
    #     total_load_energy=0.0
     
    #     for facet in Facets_dict.keys():
    #         # print facet
    #         nodes_idx = Facets_dict[facet]

    #         i = nodes_idx[0]
    #         j = nodes_idx[1]

    #         pi = np.array([nodes_prime[2*i],nodes_prime[(2*i)+1]])
    #         pj = np.array([nodes_prime[2*j],nodes_prime[(2*j)+1]])

    #         # --- Deformed length of facet:
    #         defrm_lngth = np.sqrt(np.sum((pi - pj)**2) )
            
    #         # --- Rest Length: 
    #         Lo = rest_lngth[facet]
    #         k = constant[facet]
          
    #         eargs = (Lo,defrm_lngth,k)
    #         total_energy += W(*eargs)
    #     # --- External Loads:

    #     if not Fext_array:
    #         total_load_energy = 0
    #     else:
    #         boundary_primes = nodes_prime[load_nodes_idx]
    #         boundary_primes_re = boundary_primes.reshape(int(len(boundary_primes)/2),2)
            
    #         # --- inner product(F, x) , x: node on boundary where F is applied 
    #         inner_products =  np.einsum('ij,ij->i', boundary_primes_re, Fext_array )
            
    #         # --- summation over all inner products:
    #         total_load_energy = np.sum(inner_products)
        
        
    #     U = total_energy - total_load_energy + phi_Ji
    #     # print("iter = " + repr(self._iter), "energy: "+repr(U))
    #     self._iter +=1
    #     return U


    # def Gradient(self, x, *args):
    #     """
    #     Input: u displacement array, *args: tuple of system's fixed parameters including:
    #             a. mesh coordinates 
    #             b. a dictionary with items(): {Index of each facet: (Indices of the nodes it connects,rest length)}
    #             c. an array with the rest legth of all favets.
    #             d. a dictionary with keys: the nodes of the mesh, and values the facets that pass through each node
    #             e. initial guess (as displacement) 
    #             f. a list with the indices of the FREE nodes
                
    #     Output: The gradient array for the given Energy() function
    #     ---------------------------------------------------------------------------------------

    #     Iteration over all mesh'es nodes. The code looks at each facet that comes through a particular node
    #     & computes the Gradient, for each direction, taking into account the index of the node of interest in each facet;
    #     this is necessary in order to decide the sign of the gradient at each node/direction.


    #     """
    
    #     dW = self._GradientDict[self._energy]
        
    #     u_array_mapped,coordinates,Facets_dict,rest_lngth,Nodes_dict, Nodes_of_cell,nodes_prime, free_nodes_idx,\
    #         Fext_array,load_nodes_idx,constant,areas = args

    #     # load_nodes = []
    #     # for i in range(0,len(load_nodes_idx),2): # !!!! out & asarray
    #     #     load_nodes.append(load_nodes_idx[i]/2)

    #     # --- Position Vector (after displacement u):
    #     nodes_prime[free_nodes_idx] = coordinates[free_nodes_idx] + x

    #     # self._u.vector().set_local(nodes_prime[self._idx_DofToVertex])
    #     # Ji = project(self._J, self._V0)
        
    #     dU = np.zeros(len(free_nodes_idx))
    #     count_node = 0
    #     for node in Nodes_dict.keys():
        
    #         # dx = 0.0
    #         # dy = 0.0
    #         # # print "Node: "+repr(node)
    #         '''
    #         if node in load_nodes:
    #             # print "pass"
    #             r = np.where(np.asarray(load_nodes) == node)[0][0]
    #             dx = -Fext_array[r][0]
    #             dy = -Fext_array[r][1]
    #         else:
    #             # print "pass"
    #             dx = 0.0; dy=0.0
    #         '''

    #         dx, dy =0., 0.
            
    #         vertices,lengths, edges = Nodes_dict[node] # vertciess paired with node
    #         counter = 0
            
    #         for vtx in vertices: 
    #             i = node # index of unkown node  
    #             j = vtx  # index of paired node
                
    #             x1, y1 = nodes_prime[2*i], nodes_prime[(2*i)+1] # unknown
    #             x2, y2 = nodes_prime[2*j], nodes_prime[(2*j)+1]

    #             Lo = lengths[counter] # rest length
                
    #             edge = edges[counter] # respective facet
    #             k = constant[edge]
    #             # if edge < 10:
    #             #     print "gradient facet & k : {}, {}".format(edge,k)
    #             gargs = (x1,y1,x2,y2,Lo,k)
    #             # ---
    #             gradient_array = dW(*gargs)
    #             dx += gradient_array[0] 
    #             dy += gradient_array[1]
                
    #             counter +=1
            
    #         pairs, volumes = Nodes_of_cell[node]

    #         counter2 = 0
    #         for pair in pairs:

    #             area = volumes[counter2]
    #             # print "Paired with: "+repr(pair)
    #             # print "Area: "+repr(area)
    #             idxnode_a = pair[0]
    #             idxnode_b = pair[1]

    #             # new positions
    #             fx0, fy0 = nodes_prime[2*node], nodes_prime[(2*node) +1]
    #             fxa, fya = nodes_prime[2*idxnode_a], nodes_prime[(2*idxnode_a) +1]
    #             fxb, fyb = nodes_prime[2*idxnode_b], nodes_prime[(2*idxnode_b) +1]

    #             aprm = np.array([fxa-fx0,fya-fy0])
    #             bprm = np.array([fxb-fx0,fyb-fy0])
    #             # print "Displacment a and b: {}, {}".format(aprm,bprm)
                
    #             # reference positions
    #             x0, y0 = coordinates[2*node], coordinates[(2*node) +1]
    #             xa, ya = coordinates[2*idxnode_a], coordinates[(2*idxnode_a) +1]
    #             xb, yb = coordinates[2*idxnode_b], coordinates[(2*idxnode_b) +1]

    #             aref = np.array([xa-x0,ya-y0])
    #             bref = np.array([xb-x0,yb-y0])
    #             # print "Initial a and b: {}, {}".format(aref,bref)
    #             # products
    #             init_cross = np.cross(aref,bref)#; print "initial cross"+repr(init_cross)
    #             def_cross = np.cross(aprm,bprm)#; print "after"+repr(def_cross)

    #             J = def_cross/init_cross; #print (J)
    #             dJ = self.dphi(J)
    #             dx += dJ*(fya-fyb)*(1.0/init_cross)*area
    #             dy += dJ*(fxb-fxa)*(1.0/init_cross)*area

    #             counter2 +=1
            
    #         nx = 2*count_node
    #         ny = 2*count_node +1    
    #         dU[nx] = dx
    #         dU[ny] = dy

    #         count_node += 1
        
    #     # gradF = np.asarray(dU)

    #     # Numerical Gradient:
    #     # args = (u_array_mapped,coordinates,Facets_dict,rest_lngth,Nodes_dict, Nodes_of_cell,nodes_prime, free_nodes_idx, 
    #     #         Fext_array,load_nodes_idx,constant,areas)
    #     # n = len(nodes_prime[free_nodes_idx])
    #     # h = 1e-7
    #     # numerDF = np.zeros(n)
    #     # for i in range(n):
    #     #     y = x.copy()
    #     #     y[i] += h
    #     #     numerDF[i] = (self.Energy(y,*args) - self.Energy(x.copy(),*args))/ h

    #     # print (np.max(np.abs(gradF - numerDF)))
    #     # return numerDF
    #     return dU


    # ---------
    
    
    # ===================================================================

    
    

    def get_coordinates(self):
        return self._coordinates

    def get_FacetsDict(self):
        return self._EdgesDict
    
    def get_connectivity(self):
        c = len(self._EdgesDict.keys())
        return c

    def get_domain(self):
        self._domain.check_domain()
        return self._domain
    
    def get_circles(self):
        return self._circles

    def get_V(self):
        return self._V
    
    def get_VertexToDof(self):
        return self._idx_VertexToDof
    
    def get_DofToVertex(self):
        return self._idx_DofToVertex

    def get_free_nodes_extend_ar(self):
        return self._free_nodes_extend
    
    def get_free_nodes_extend_bycell(self):
        return self._free_nodes_extend_per_cell
    
    def get_init_cross_product(self):
        return self._init_cross

    def save_boundary_indices(self,Vo = None,name_file = None):
        '''
        returns the double-indices of boundaries' indices
        i.e. for both coordinates
        '''
        circles = self._circles
        domain = self._domain

        cell_idx = []
        for cell in circles:
            cell_idx.append(self.extract_boundary_index(cell, Vo))

        domain_nodes = self.extract_boundary_index(domain, Vo)
        if len(cell_idx) == 2:
            D = dict([ ('domain', np.asarray(domain_nodes)), ('right',np.asarray(cell_idx[0])), ('left',np.asarray(cell_idx[1]))])
        else:
            D = dict([ ('domain',np.asarray(domain_nodes)), ('right',np.asarray(cell_idx[0]))])

        if name_file is None:
            return D
        else:  
            import pickle
            with open(name_file, 'wb') as fp:
                pickle.dump(D, fp)
    
    
    def save_boundary_fibers(self, e_coor = None,Vo = None,name_file = None):
        '''
        returns the single index of each fiber on each boundary
        (order same as appears in main fiber-index EdgesDict from connectivity matrix)
        '''
        circles = self._circles
        domain = self._domain

        if e_coor is None:
            e_coor  = self._e_coor_indx

        cell_idx = []
        for cell in circles:
            double_idx = self.extract_boundary_index(cell, Vo)
            cell_idx.append(self.return_idx_of_vertices(double_idx))
        domain_nodes_double = self.extract_boundary_index(domain, Vo)
        dmn_idx = self.return_idx_of_vertices(domain_nodes_double)

        # =====
        if name_file is None:
            # cell_idx.append(dmn_idx); print(len(cell_idx)) # !!!! (un)comment whether u want domain idx or not in the returning list !!!! 
            All_idx = [item for sublist in cell_idx  for item in sublist]#;print(All_idx)
            mylist = [np.isin(np.array([i,j]), All_idx) for i,j in zip(e_coor[::2], e_coor[1::2])]#;print(mylist)
            fibers_bndr_idx = [i for i in range(len(mylist)) if all(mylist[i] == True)]
            return np.asarray(fibers_bndr_idx)
        else:  ## !!!! to be added !!!!
            import pickle
            with open(name_file, 'wb') as fp:
                pickle.dump(D, fp)
    
    
    def get_mesh(self):
        return self._mesh
    
    def get_rest_lengths(self):
        return self._restlngth_ar

    def get_idx_facets(self):
        return self._e_coor_indx

    def get_refined_fibers_idx(self):
        return self._refined_fibers_idx
# ========================================================================
# ========================================================================

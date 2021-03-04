import numpy as np
from collections import OrderedDict
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, ConvexHull, distance_matrix

from .base import createTINgraph, get_vor_area
from ..utils import make_parallel, round_points


class JetPersistance:
    '''
    compute persistence of Jets
    '''
    def __init__(self):
        super(JetPersistance, self).__init__()    

    def _b0_diagram(self, jet_4p, zeta_type='zeta', R=0.6):

        r'''
        compute the persistance diagram for connected components 
        '''        

        jet_pt = jet_4p.sum().pt
        points = np.vstack((jet_4p.eta, jet_4p.phi)).T
        points = round_points(points)
        ## descending sorted w.r.t zeta
        if zeta_type == 'zeta':
            zeta = np.array([ jet_4p[i].pt / jet_pt for i in range(len(jet_4p)) ])
            idx = np.argsort(zeta)[::-1]
        elif zeta_type == 'pt_density':            
            zeta = np.array([ jet_4p[i].pt / jet_pt for i in range(len(jet_4p)) ])
            area = get_vor_area(points, R=R)
            zeta = zeta / area
            idx = np.argsort(zeta)[::-1]
        zeta = zeta[idx]
        jet_4p = jet_4p[idx]
        points = points[idx]

        ## create DT graph
        graph = createTINgraph(points)                
    
        jet_branches = OrderedDict()

        for i, cc in enumerate(zeta):
            ## super-level graph
            nodes2del = list( np.arange(len(jet_4p))[ (1+i): ] ) 
            H = graph.copy()
            H.remove_nodes_from(nodes2del)
            comp_ll = [ c for c in nx.connected_components(H) ]
            ## compute [b, d] pairs, where b >= d
            if len(comp_ll) > 0:
                for compset in comp_ll:
                    compset = sorted(list(compset))
                    if str(compset[0]) not in jet_branches:
                        if i < len(zeta) - 1:
                            jet_branches[str(compset[0])] = [ cc, zeta[i+1] ]
                        else:
                            jet_branches[str(compset[0])] = [ cc, cc ]
                    elif str(compset[0]) in jet_branches:
                        if i < len(zeta) - 1:
                            jet_branches[str(compset[0])][1] = zeta[i+1] 
                        else:
                            jet_branches[str(compset[0])][1] = cc                        
        return np.array( [ jet_branches[key] for key in jet_branches ] ) 

    def _b1_diagram(self, jet_4p, zeta_type='zeta', R=0.6):
        r'''
        compute persistence diagrams for holes
        '''

        jet_pt = jet_4p.sum().pt
        points = np.vstack((jet_4p.eta, jet_4p.phi)).T
        points = round_points(points)
        ## ascending sorted w.r.t zeta
        if zeta_type == 'zeta':
            zeta = np.array([ jet_4p[i].pt / jet_pt for i in range(len(jet_4p)) ])
            idx = np.argsort(zeta)[::-1]
        elif zeta_type == 'pt_density':            
            zeta = np.array([ jet_4p[i].pt / jet_pt for i in range(len(jet_4p)) ])
            area = get_vor_area(points, R=R)
            zeta = zeta / area
            idx = np.argsort(zeta)[::-1]
        zeta = zeta[idx]
        jet_4p = jet_4p[idx]
        points = points[idx]

        ## create DT graph
        graph = createTINgraph(points, addVirtual=True)  

        anchor_dic = {}        

        for i, cc in enumerate(zeta):
            nodes2del = list(np.arange(len(jet_4p))[(1+i):])
            H = graph.copy()
            H.remove_nodes_from(nodes2del)
            comp_ll = [c for c in nx.connected_components(H)]
            ## compute [b, d] pairs, where b >= d
            if len(comp_ll) > 0:
                for compset in comp_ll:
                    compset = sorted(list(compset))
                    if (str(list(compset)[0]) not in anchor_dic) and (str(list(compset)[0])!=str(-1)):
                        anchor_dic[str(list(compset)[0])] = [zeta[i+1], cc]
                    elif str(list(compset)[0]) in anchor_dic: 
                        anchor_dic[str(list(compset)[0])][0] = zeta[i+1] if cc<zeta[-1] else zeta[-1]                    
        return np.array( [anchor_dic[key] for key in anchor_dic] )

    #@staticmethod
    def compute_persistence(self, jet_4ps, case=['b0', 'b1'], zeta_type='zeta', R=0.6, n_jobs=-1):
        '''
        compuet persistence diagrams for a list of jet 4-momenta
        '''
        result = {}
        for key in case:
            if key == 'b0':
                result[key] = make_parallel(self._b0_diagram, n_jobs=n_jobs, zeta_type=zeta_type, R=R)(jet_4ps)
            elif key == 'b1':
                result[key] = make_parallel(self._b1_diagram, n_jobs=n_jobs, zeta_type=zeta_type, R=R)(jet_4ps)
        return result

    def _PI(self, diagram, pixels=(20, 20)):
        '''
        compute persistence image
        
        `diagram` a list of [b, d] pairs
        '''
        nx, ny = pixels
        if len(diagram) == 0:
            return np.zeros((nx, ny))
        diagram = np.array(diagram)
        ## convert [b, d] -> [b, b / d]
        diagram[:, 1] = diagram[:, 0] / diagram[:, 1]

        pass

    def _PL(self):
        '''
        compute persistence landscapes
        '''
        pass



class ML_JetPersistance(JetPersistance):
    '''
    compute ML inputs for persistence related variables
    '''
    def __init__(self):
        super(ML_JetPersistance, self).__init__()
    
    def _get_b0_ml_inputs(self, jet_4p, zeta_type='zeta', R=0.6, show_history=False):
        '''
        compute ml inputs for connected components as well the jet branches evolving:

        if `show_history=False`:
            returns: inputs: [ log(b), log(d), idx_merged, log(HT_branch), log(HT_branch^merged) ] for all pairs

        if `show_history=True`:
            returns: `history`
        '''

        jet_pt = jet_4p.sum().pt
        jet_ht = np.sum(jet_4p.pt)
        points = np.vstack((jet_4p.eta, jet_4p.phi)).T
        points = round_points(points)
        ## descending sorted w.r.t zeta
        if zeta_type == 'zeta':
            zeta = np.array([ jet_4p[i].pt / jet_pt for i in range(len(jet_4p)) ])
            idx = np.argsort(zeta)[::-1]
        elif zeta_type == 'pt_density':            
            zeta = np.array([ jet_4p[i].pt / jet_pt for i in range(len(jet_4p)) ])
            area = get_vor_area(points, R=R)
            zeta = zeta / area
            idx = np.argsort(zeta)[::-1]

        zeta = zeta[idx]
        jet_4p = jet_4p[idx]
        points = points[idx]        

        ## create DT graph
        graph = createTINgraph(points, addVirtual=False)

        ## record particles of each connected component at each zeta cut
        cut_history = OrderedDict() 
        for i, cc in enumerate(zeta):
            ## super-level graph
            nodes2del = list(np.arange(len(jet_4p))[ (i+1): ])
            H = graph.copy()
            H.remove_nodes_from(nodes2del)
            comp_ll = [c for c in nx.connected_components(H)]            
            cut_history[cc] = OrderedDict()
            if len(comp_ll) > 0:
                for compset in comp_ll:
                    compset = sorted(list(compset))                 
                    cut_history[cc][compset[0]] = compset

        ## get components for each jet branch and persistence diagrams
        jet_branches = OrderedDict()
        persistence = OrderedDict()           
        for i, cc in enumerate(cut_history.keys()):
            for key in cut_history[cc]:
                compset = cut_history[cc][key]
                if compset[0] not in jet_branches:
                    jet_branches[compset[0]] = []                    
                jet_branches[compset[0]].append(compset)
                if compset[0] not in persistence:
                    if len(zeta) == 1:
                        persistence[compset[0]] = [ cc, cc ]
                    if i < len(zeta) - 1:
                        persistence[compset[0]] = [ cc, zeta[i + 1] ]
                    else:
                        persistence[compset[0]] = [ cc, cc ]
                elif compset[0] in persistence:
                    if i < len(zeta) - 1:
                        persistence[compset[0]][1] = zeta[i+1]
                    else:
                        persistence[compset[0]][1] = cc

        ## find the idx of the connected component that the this one merged to.
        id_merge = OrderedDict()
        for i, cc in enumerate(zeta):
            if i < (len(zeta) - 1):
                for key in cut_history[cc]:
                    if key not in cut_history[ zeta[i+1] ]:
                        for key_i in cut_history[ zeta[i+1] ]:
                            if int(key) in cut_history[ zeta[i+1] ][key_i]:
                                id_merge[key] = key_i
        id_merge[0] = 0

        if show_history:
            history = {}
            history['cut_history'] = cut_history
            history['persistence'] = persistence
            history['id_merge'] = id_merge
            return history

        else:
            ml_inputs = []        
            if len(zeta) == 1:
                b, d = np.log(zeta[0]), np.log(zeta[0])
                id_s, id_m = 0, 0
                HT_s, HT_m = 1., 1.
                ml_inputs.append(np.array([b, d, id_m, HT_s, HT_m]))   
                return ml_inputs

            for key in persistence:
                ## log(birth), log(death)
                b, d = np.log(persistence[key][0]), np.log(persistence[key][1])
                ## idx for each jet branch and which one it merged to
                id_s, id_m = list(persistence.keys()).index(key), list(persistence.keys()).index(id_merge[key])                
                ## compute HT for the branch at the moment it dies and the branch it merged to.
                HT_s = np.log(np.sum(jet_4p[jet_branches[key][-1]].pt) / jet_ht)
                merge_cc_idx = int(list(zeta).index(persistence[key][1]) - 1)

                try:
                    merge_comp = cut_history[zeta[merge_cc_idx]][id_merge[key]]
                except:
                    try:
                        merge_comp = cut_history[zeta[merge_cc_idx + 1]][id_merge[key]]
                    except:
                        for cc in zeta[::-1]:
                            ## find the hardest moment that the branches merged to 
                            if id_merge[key] in cut_history[cc]:
                                merge_comp = cut_history[cc][id_merge[key]]
                if not merge_comp:
                    ## if not find the component that it merges to, assign this branch merges to the `0` branch 
                    ## and find the connected components at the death moment
                    try:
                        merge_comp = cut_history[persistence[key][1]][0]
                    except:
                        merge_comp = [0]                        
                HT_m = np.log(np.sum(jet_4p[merge_comp].pt) / jet_ht)
                ml_inputs.append(np.array([b, d, id_m, HT_s, HT_m]))  
                   
            ml_inputs = np.array(ml_inputs)
            if len(ml_inputs) > 1:
                return ml_inputs[np.argsort(ml_inputs[:, 0] - ml_inputs[:, 1])[::-1]]
            else:
                return ml_inputs


    def _get_b1_ml_inputs(self, jet_4p, zeta_type='zeta', R=0.6):
        '''
        compute the ml inputs for holes
        '''

        jet_pt = jet_4p.sum().pt
        jet_ht = np.sum(jet_4p.pt)        
        points = np.vstack((jet_4p.eta, jet_4p.phi)).T
        points = round_points(points)
        ## ascending sorted w.r.t zeta
        if zeta_type == 'zeta':
            zeta = np.array([ jet_4p[i].pt / jet_pt for i in range(len(jet_4p)) ])
            idx = np.argsort(zeta)[::-1]
        elif zeta_type == 'pt_density':            
            zeta = np.array([ jet_4p[i].pt / jet_pt for i in range(len(jet_4p)) ])
            area = get_vor_area(points, R=R)
            zeta = zeta / area
            idx = np.argsort(zeta)[::-1]
        zeta = zeta[idx]
        jet_4p = jet_4p[idx]
        points = points[idx]

        ## create DT graph
        graph = createTINgraph(points, addVirtual=True) 
        orig_graph = createTINgraph(points, addVirtual=False)

        persistence = OrderedDict()
        hole_weight = OrderedDict()
        
        for i, cc in enumerate(zeta):
            nodes2del = list(np.arange(len(zeta))[(1+i):])
            orig_nodes2del = list(np.arange(len(zeta))[:(1+i)])
            H = graph.copy()
            H.remove_nodes_from(nodes2del)
            orig_H = orig_graph.copy()
            orig_H.remove_nodes_from(orig_nodes2del)
            comp_ll = [ c for c in nx.connected_components(H) ]
            if len(comp_ll) > 0:
                for compset in comp_ll:
                    compset = sorted(list(compset))
                    if compset[0] != -1:
                        neigh_nodes = set([n for tmp_node in compset for n in orig_graph.neighbors(tmp_node)])
                        neigh_nodes = neigh_nodes & set(orig_H.nodes()) - set(compset)
                        neigh_nodes = sorted(list(neigh_nodes))
                        if compset[0] not in persistence:
                            persistence[compset[0]] = [ zeta[i+1], cc ] if i<(len(zeta)-1) else [cc, cc]
                            hole_weight[compset[0]] = [
                                np.sum(jet_4p[neigh_nodes].pt) / jet_ht, 
                                np.sum(jet_4p[neigh_nodes].pt) / jet_ht
                                ]
                        elif compset[0] in persistence:
                            persistence[compset[0]][0] = zeta[i+1] if cc<zeta[-1] else zeta[-1]
                            hole_weight[compset[0]][0] = np.sum(jet_4p[neigh_nodes].pt) / jet_ht

        pers_pairs = np.array([persistence[key] for key in persistence])
        if len(pers_pairs) > 1:
            weights = np.array([hole_weight[key] for key in hole_weight])
            ml_inputs = np.concatenate((np.log(pers_pairs), np.log(weights)), axis=1)
            if len(ml_inputs) > 1:
                return ml_inputs[np.argsort(ml_inputs[:, 0] - ml_inputs[:, 1])[::-1]]
            else:
                return ml_inputs
        return []            

    #@staticmethod
    def get_ml_inputs(self, jet_4ps, case=['b0', 'b1'], zeta_type='zeta', R=0.6, n_jobs=-1):
        '''
        compute ml inputs parallelly
        '''
        result = {}
        for key in case:
            if key == 'b0':
                result[key] = make_parallel(self._get_b0_ml_inputs, n_jobs=n_jobs, zeta_type=zeta_type, R=R)(jet_4ps)
            elif key == 'b1':
                result[key] = make_parallel(self._get_b1_ml_inputs, n_jobs=n_jobs, zeta_type=zeta_type, R=R)(jet_4ps)
        return result
        
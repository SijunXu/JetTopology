import numpy as np
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, ConvexHull, distance_matrix
from scipy.spatial import cKDTree
from numba import jit

'''
A pipline for Topological Data Analysis with inputs of a set of particles in a jet
inlcudes:
betti numbers for superset / subset
persistant homology 
downstream information for machine learning
'''


def createTINgraph(points, addVirtual=False):
    
    r'''
    create Delaunay triangulated graph for a jet
    '''

    ## if the number of particles in a jet is less than 3, then return fully connected graph
    rand_points = np.random.uniform(-1e-6, 1e-6, size=points.shape)
    points += rand_points
    if len(points) < 3:
        if len(points) == 1:
            graph = nx.Graph()
            graph.add_node(0)
            return graph
        edges = set((i, j) for i in range(len(points)) for j in range(len(points)) if i < j)
        return nx.Graph(edges)
    ## if 
    try:    
        TIN = Delaunay(points)
    except:
        print( points + ' cannot be triangulated with '+str(len(points))+' particles, returns a fully connected graph' )
        edges = set((i, j) for i in range(len(points)) for j in range(len(points)) if i < j)
        return nx.Graph(edges)  

    graph = nx.Graph()    
    for path in TIN.vertices:
        nx.add_path(graph, path)    
    if addVirtual:
        edges = set()
        hull = ConvexHull(points)
        vertex = hull.vertices
        for idx in vertex:
            edges.add((-1, idx))        
        graph.add_edges_from(list(edges))
    if not addVirtual and len(points) > len(graph.nodes()):
        return createTINgraph(points, addVirtual=False)
    if addVirtual and len(points)+1 > len(graph.nodes()):
        return createTINgraph(points, addVirtual=True)
    return graph


def compute_dist(p4, p=0):
    r'''
    compute distance between particles 
    '''
    array = np.stack([p4.eta, p4.phi]).T
    delta = distance_matrix(array, array)    
    dist = np.zeros((len(p4), len(p4)))
    for i in range(len(p4)):
        for j in range(len(p4)):                
            if i < j:
                dist[i, j] = min(p4.pt[i] ** (2*p), p4.pt[j] ** (2*p)) * delta[i, j]
    return np.maximum( dist, dist.transpose() )

def create_kNN_graph(dist, k=3):
    r'''
    craete k-NN graph based on N * N distance 
    '''
    edges = set()
    n_p = len(dist)
    k = min(n_p - 1, k)
    kNN_idx = np.argsort(dist, axis=1)[:, 1:(k+1)]
    for i, idx in enumerate(kNN_idx):
        for j in idx:
            edges.add( (i, j) )
    # for i in range(len(dist)):
    #     mask = [i!=i_col for i_col in range(len(dist))]
    #     dist_i = dist[i, mask]
    #     kNN_idx = np.argpartition(dist_i, k)[:min(len(n_p), k)]
    #     for idx in kNN_idx:
    #         edges.add( (i, idx) )            
    graph = nx.Graph(list(edges))
    return graph
    
        

def get_vor_area(points, R=0.4, n_estimates=500000):

    r'''
    get the Voronoi area for each particle within a jet radius of R, 
    randomly sampling `n_estimates` numboer of partiles to estimate 

    input:
        `points`: coordinates of particles
        `R`: radius 
        `n_estimates`: number of estimates 
    returns: 
        Vor areas / jet_area
    '''

    alpha = 2.0 * np.pi * np.random.random(n_estimates)
    r = R * np.random.random(n_estimates)
    test_points =  np.vstack([r * np.cos(alpha), r * np.sin(alpha)]).T
    voronoi_kdtree = cKDTree(points)
    _, test_point_regions = voronoi_kdtree.query(test_points)
    unique_index, unique_counts = np.unique(test_point_regions, return_counts=True)
    area = np.zeros((len(points)))
    area[unique_index] = (unique_counts / n_estimates) #* np.pi * R ** 2
    ## avoid zeros values for each cell
    area[~unique_index] += 1e-5
    return area


from ..utils import make_parallel, round_points

class Betti:

    r'''
    compute Betti numbers given 4-vectors of a jet
    '''

    def __init__(self):        
        super().__init__()    
    
    def _sampling(self, bins=50, cutRange=(1e-2, 1), logSpace=False, log_base=np.e):
        r'''
        sampling 1d points within ``cutRange``
        '''
        if logSpace:
            a, b = np.log(cutRange[0]) / np.log(log_base), np.log(cutRange[1]) / np.log(log_base)
            cuts = np.logspace(a, b, bins, base=log_base)
        else:
            cuts = np.linspace(cutRange[0], cutRange[1], bins)
        return cuts

    def _graph2beta(self, graph, hull):
        r'''
        calculate betti numbers given a graph and the convex hull
        $\beta_0, \beta_1, \bar{\beta_0}, \bar{\beta_1}$
        '''
        beta = np.zeros(2)  
        beta[0] = nx.number_connected_components(graph)
        cc_list = nx.connected_components(graph)
        nb_cc_bry = 0
        for xx in cc_list:
            for idx_cc in xx:
                if idx_cc in hull.vertices:
                    nb_cc_bry += 1
                    break
        beta[1] = nb_cc_bry
        return beta  
    
    def _filtering_kNN_beta(self, X_4p, k=2, p=0, **sample_params):
        r'''
        compute b_0 for kNN based graph for superset
        zeta = pT / pT_jet
        '''
        cuts = self._sampling(**sample_params)
        n_points = len(X_4p)
        bins = sample_params['bins']
        beta = np.zeros(bins)
        dist = compute_dist(X_4p, p=p)
        graph = create_kNN_graph(dist, k=k)

        zeta = X_4p.pt / X_4p.sum().pt

        for i, cc in enumerate(cuts):
            nodes2del = list(np.arange(n_points)[zeta <= cc])
            if i == 0:
                tmp_nodes2del = nodes2del
                H = graph.copy()
                H.remove_nodes_from(nodes2del)
                beta[i] = nx.number_connected_components(H)
            else:
                if (len(tmp_nodes2del)==len(nodes2del)):
                    beta[i] = beta[i-1]
                    tmp_nodes2del = nodes2del
                else:
                    H = graph.copy()
                    H.remove_nodes_from(nodes2del)                
                    beta[i] = nx.number_connected_components(H)
                    tmp_nodes2del = nodes2del
        return beta                

    def _filtering_zeta(self, graph, X_4p, zeta_type='zeta', case='dual', R=0.6, **sample_params):
        r'''
        filtering DT graph w.r.t defined `zeta` by `zeta_type`

        `zeta_type=='zeta'`, zeta = pT / pT_jet

        `zeta_type=='pt_density'`, zeta = (pT / pT_jet) / (area / jet_area)

        '''
        cuts = self._sampling(**sample_params)
        bins = sample_params['bins']
        beta = np.zeros((bins, 2))

        points = np.vstack((X_4p.eta, X_4p.phi)).T
        points = round_points(points)
        n_points = len(points)
        hull = ConvexHull(points)        

        if zeta_type == 'zeta':
            zeta = X_4p.pt / X_4p.sum().pt
        elif zeta_type == 'pt_density':
            area = get_vor_area(points, R=R)
            zeta = X_4p.pt / X_4p.sum().pt 
            zeta = zeta / area

        for i, cc in enumerate(cuts):
            if case=='dual':
                nodes2del = list(np.arange(n_points)[zeta > cc])
            else:
                nodes2del = list(np.arange(n_points)[zeta <= cc]) 
            if i==0:
                tmp_nodes2del = nodes2del
                H = graph.copy()
                H.remove_nodes_from(nodes2del)
                beta[i] = self._graph2beta(H, hull)
            else:
                if (len(tmp_nodes2del)==len(nodes2del)):
                    beta[i] = beta[i-1]
                    tmp_nodes2del = nodes2del
                else:
                    H = graph.copy()
                    H.remove_nodes_from(nodes2del)                
                    beta[i] = self._graph2beta(H, hull)   
                    tmp_nodes2del = nodes2del
        return beta

    #@staticmethod
    def _compute_beta(self, X_4p, zeta_type='zeta', R=0.6, **sample_params):
        r'''
        compute betti numbers for each jet, given 4p of particles, 
        cut on threshold of pT fraction(zeta) or collinear angle(dr)
        '''
        jet = X_4p.sum()
        points = np.vstack((X_4p.eta - jet.eta, X_4p.phi - jet.phi)).T
        graph = createTINgraph(points)        

        ## compute betti number for super-level and sub-level
        beta_a = self._filtering_zeta(graph, X_4p, case='not_dual', zeta_type=zeta_type, R=R, **sample_params)
        beta_b = self._filtering_zeta(graph, X_4p, case='dual', zeta_type=zeta_type, R=R, **sample_params)    

        beta[:, 0] = beta_a[:, 0]
        beta[:, 2] = beta_b[:, 0]
        beta[:, 3] = beta_a[:, 0] - beta_a[:, 1]
        beta[:, 1] = beta_b[:, 0] - beta_b[:, 1]
        beta = np.stack(
            [
                beta_a[:, 0],
                beta_b[:, 0] - beta_b[:, 1],
                beta_b[:, 0],
                beta_a[:, 0] - beta_a[:, 1]
            ]
        ).T
        return beta        

    #@staticmethod
    def compute_beta(self, X_4ps, zeta_type='zeta', R=0.6, n_jobs=-1, **sample_params):
        r'''
        compute betti numbers for a list of jets with 4-momenta
        '''        
        result = make_parallel(self._compute_beta, n_jobs=n_jobs, zeta_type=zeta_type, R=R, **sample_params)(X_4ps)
        return np.array(result)

    def compute_kNN_beta(self, X_4ps, k=2, p=0, n_jobs=-1, **sample_params):
        r'''
        computer betti numbers for a list of jets with 4-momenta for superset
        '''
        result = make_parallel(self._filtering_kNN_beta, n_jobs=n_jobs, k=k, p=p, **sample_params)(X_4ps)
        return np.array(result)
    
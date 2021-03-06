import numpy as np

def round_phi(data):
    r'''
    rotate the phi of data within (0, 2 * pi)    
    '''
    data[:, :, 2] = data[:, :, 2] - 2 * np.pi * np.floor(data[:, :, 2] / (np.pi * 2)) 
    return data

def round_points(points):
    r'''
    round points to make phi in range (-pi, pi]
    '''
    points[:, 1][points[:, 1] > np.pi] -= 2 * np.pi
    points[:, 1][points[:, 1] <= -np.pi] += 2 * np.pi
    return points


import uproot_methods
from scipy.spatial import distance_matrix

from . import make_parallel

class IRC_cut:
    '''
    merge collinear particles and remove soft particles
    '''
    def __init__(self):
        super().__init__()    

    def _merge(self, X_4p, dRmin=0.01, soft=False):  
        r'''
        merge particles until all particles with a between distance greater than `dRmin`
        with a priority to mearge hardest/softest pairs with distance less than `dRmin`
        '''
        if soft:
            X_4p = X_4p[np.argsort(X_4p.pt)]
        else:
            X_4p = X_4p[np.argsort(X_4p.pt)[::-1]]
        points = np.vstack((X_4p.eta, X_4p.phi)).T
        mx = distance_matrix(points, points)
        idx = np.where( mx < dRmin )
        nb_pairs = len(idx[0])
        if nb_pairs > 0:
            pair2merge = []
            for i in range(nb_pairs):
                if idx[0][i] < idx[1][i]:
                    if i==0:
                        pair2merge.append(idx[0][i])
                        pair2merge.append(idx[1][i])
                    else:
                        if (idx[0][i] in pair2merge) or (idx[1][i] in pair2merge):
                            continue
                        else:
                            pair2merge.append(idx[0][i])
                            pair2merge.append(idx[1][i])
            arr_4p = np.vstack((X_4p.x, X_4p.y, X_4p.z, X_4p.t)).T
            for i in range(len(pair2merge)//2):
                arr_4p[pair2merge[2*i]] += arr_4p[pair2merge[2*i+1]]
                arr_4p[pair2merge[2*i+1]] = 0
            arr_4p = arr_4p[arr_4p[:, 3]>0]
            return uproot_methods.TLorentzVectorArray.from_cartesian(arr_4p[:, 0], arr_4p[:, 1], 
                                                                    arr_4p[:, 2], arr_4p[:, 3])
        else:
            return X_4p        

    def _merge_near(self, X_4p, dRmin=0.01):
        r'''
        merge particles until all particles with a between distance greater than `dRmin`
        with a priority to mearge nearest pairs with distance less than `dRmin`
        '''
        points = np.vstack((X_4p.eta, X_4p.phi)).T
        mx = distance_matrix(points, points)
        idx = np.where( mx < dRmin )
        nb_pairs = len(idx[0])

        if nb_pairs > 0:
            dist_ll = np.array([mx[idx[0][j], idx[1][j]] for j in range(nb_pairs) if idx[0][j] < idx[1][j]])
            idx_ll = np.array([[idx[0][j], idx[1][j]] for j in range(nb_pairs) if idx[0][j] < idx[1][j]])
            sort_idx = np.argsort(dist_ll)
            dist_ll = dist_ll[sort_idx]
            idx_ll = idx_ll[sort_idx]
            pair2merge = []
            for i, id_p in enumerate(idx_ll):
                if i==0:
                    pair2merge.append(id_p[0])
                    pair2merge.append(id_p[1])
                else:
                    if (id_p[0] in pair2merge) or (id_p[1] in pair2merge):
                        continue
                    else:
                        pair2merge.append(id_p[0])
                        pair2merge.append(id_p[1])
            arr_4p = np.vstack((X_4p.x, X_4p.y, X_4p.z, X_4p.t)).T
            for i in range(len(pair2merge)//2):
                arr_4p[pair2merge[2*i]] += arr_4p[pair2merge[2*i+1]]
                arr_4p[pair2merge[2*i+1]] = 0
            arr_4p = arr_4p[arr_4p[:, 3]>0]
            return uproot_methods.TLorentzVectorArray.from_cartesian(arr_4p[:, 0], arr_4p[:, 1],
                                                                    arr_4p[:, 2], arr_4p[:, 3])
        else:
            return X_4p

    def _remove_soft(self, X_4p, zeta_min=1e-2):
        r'''
        remove soft particles with pT / pT_jet less than zeta_min
        '''
        jet_pt = X_4p.sum().pt
        return X_4p[ X_4p.pt / jet_pt > zeta_min ]  
    
    #@staticmethod
    def _IRC_safe(self, X_4p, dRmin=0.01, zeta=5e-3, soft=False, near=False, max_par=200):

        min_mx = dRmin/2.
        while min_mx <= dRmin:
            if near:
                X_4p = self._merge_near(X_4p, dRmin)
            else:
                X_4p = self._merge(X_4p, dRmin, soft=soft)

            points = np.vstack((X_4p.eta, X_4p.phi)).T
            mx = distance_matrix(points, points)
            mask = np.ones(mx.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            if len(mask)<=1:
                min_mx = np.inf
            else:
                min_mx = mx[mask].min()

        X_4p = self._remove_soft(X_4p, zeta)
        nb_par = len(X_4p)
        jet_particle = np.zeros((max_par, 4), dtype=np.float32)
        jet_particle[:nb_par] = np.vstack((X_4p.pt, X_4p.eta, X_4p.phi, X_4p.mass)).T
        return jet_particle    
        # return X_4p

    #@staticmethod
    def process(self, X_4ps, dRmin=1e-2, zeta=1e-2, near=True, soft=False, max_par=200):
        '''
        parallelly make technical IRC safe cut on jets 
        '''
        result = make_parallel(self._IRC_safe, dRmin=dRmin, zeta=zeta, near=near, soft=soft, max_par=max_par)(X_4ps)
        return np.array(result)
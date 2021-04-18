import numpy as np
from pyjet import cluster, DTYPE_PTEPM
from scipy.spatial import distance_matrix

import JetTopology.utils as utils

class JetObs:
    '''
    compute jet substructure observables: generalised angularities, ECFs and N-subjettiness
    '''
    def __init__(self):        
        super().__init__()

    def _angularity(self, jet_p4, beta, k, R=0.6):
        if len(jet_p4) <= 1:
            return 0
        z = jet_p4.pt / np.sum(jet_p4.pt)
        ang = ((jet_p4.eta - jet_p4.sum().eta) ** 2 + (jet_p4.phi - jet_p4.sum().phi) **2) ** .5 / R
        return np.sum( (z ** k) * (ang ** beta) )

    def _2ecf(self, jet_p4, beta):        
        z = jet_p4.pt / np.sum(jet_p4.pt)
        ecf = 0.0
        pos = np.vstack([jet_p4.eta, jet_p4.phi]).T
        dist = distance_matrix(pos, pos)
        for i in range(len(z)):
            for j in range(len(z)):
                if i < j:
                    #ang_ij = ( (jet_p4[i].eta - jet_p4[j].eta) ** 2 + (jet_p4[i].phi - jet_p4[j].phi) ** 2 ) ** .5
                    ecf += z[i] * z[j] * ( dist[i, j] ** beta )
        return ecf 

    def _cluster_jets(self, jet_p4, n_jet, max_R):        
        pseudojets_input = np.zeros(len(jet_p4), dtype=DTYPE_PTEPM)
        for i, p4 in enumerate(jet_p4):
            pseudojets_input[i]['pT'] = p4.pt
            pseudojets_input[i]['eta'] = p4.eta
            pseudojets_input[i]['phi'] = p4.phi
            pseudojets_input[i]['mass'] = p4.mass
        sequence = cluster(pseudojets_input, R=max_R, p=1)
        jets = sequence.exclusive_jets(n_jet) ## decluster
        return jets
        

    def _Njettiness(self, jet_p4, N=2, beta=1, R=0.6):
        '''
        using exclusive kt to cluster subjets
        '''
        if len(jet_p4) < N:
            return 0.0
        jets = self._cluster_jets(jet_p4, n_jet=N, max_R=R)
        #points = np.vstack([jet_p4.eta, jet_p4.phi]).T
        #subjets_pos = np.zeros((N, 2))
        #points = utils.round_points(points)
        #subjets_pos = utils.round_points(points)
        dists = np.zeros((len(jet_p4), N))
        for j in range(len(jet_p4)):
            for i in range(N):            
                dists[j, i] = jet_p4[j].delta_r(jets[i])
                #subjets_pos[i] = jets[i].eta, jets[i].phi
        #subjets_pos = np.vstack([jets.eta, jets.phi])
        #dists = distance_matrix(points, subjets_pos) ** beta
        dists = dists ** beta
        z = jet_p4.pt / sum(jet_p4.pt)
        tau  = sum(z * np.min(dists, axis=1)) / (R ** beta)
        return tau

    def angularity(self, jet_p4s, beta, k, R=0.6):
        return np.array(utils.make_parallel(self._angularity, beta=beta, k=k, R=R)(jet_p4s))

    def ecf(self, jet_p4s, beta):
        return np.array(utils.make_parallel(self._2ecf, beta=beta))(jet_p4s)

    def Njettiness(self, jet_p4s, N=2, beta=1, R=0.6):
        return np.array( utils.make_parallel(self._Njettiness, N=N, beta=beta, R=R)(jet_p4s) )

        
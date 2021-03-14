import numpy as np
from . import make_parallel

from pyjet import cluster, DTYPE_PTEPM
from scipy.spatial import distance_matrix

class JetObs:
    '''
    compute jet substructure observables: generalised angularities, ECFs and N-subjettiness
    '''
    def __init__(self):        
        super().__init__()

    def _angularity(self, jet_p4, beta, k, R=0.6):
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

    def _cluster_jets(self, jet_p4, n_jet):        
        pseudojets_input = np.zeros(len(jet_p4), dtype=DTYPE_PTEPM)
        for i, p4 in enumerate(jet_p4):
            pseudojets_input[i]['pT'] = p4.pt
            pseudojets_input[i]['eta'] = p4.eta
            pseudojets_input[i]['phi'] = p4.phi
            pseudojets_input[i]['mass'] = p4.mass
        sequence = cluster(pseudojets_input, algo='ee_kt')
        jets = sequence.exclusive_jets(n_jet)
        return jets
        

    def _Njettiness(self, jet_p4, N=2, beta=1, R=0.6):
        '''
        using exclusive kt to cluster subjets
        '''
        if len(jet_p4) < N:
            return 0.0
        jets = self._cluster_jets(jet_p4, n_jet=N)
        points = np.vstack([jet_p4.eta, jet_p4.phi]).T
        subjets_pos = np.zeros((N, 2))
        for i in range(N):
            subjets_pos[i] = jets[i].eta, jets[i].phi
        #subjets_pos = np.vstack([jets.eta, jets.phi])
        dists = distance_matrix(points, subjets_pos) ** beta
        z = jet_p4.pt / sum(jet_p4.pt)
        tau  = sum(z * np.min(dists, axis=1)) / (R ** beta)
        return tau

    def angularity(self, jet_p4s, beta, k, R=0.6):
        return np.array(make_parallel(self._angularity, beta=beta, k=k, R=R)(jet_p4s))

    def ecf(self, jet_p4s, beta):
        return np.array(make_parallel(self._2ecf, beta=beta))(jet_p4s)

    def Njettiness(self, jet_p4s, N=2, beta=1, R=0.6):
        return np.array( make_parallel(self._Njettiness, N=N, beta=beta, R=R)(jet_p4s) )

        
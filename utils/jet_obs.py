import numpy as np

class JetObs:
    '''
    compute jet substructure observables: generalised angularities, ECFs and N-subjettiness
    '''
    def __init__(self):        
        super().__init__()

    def _angularity(self, jet_p4, beta, k, R=0.6):
        z = jet_p4.pt / np.sum(jet_p4.pt)
        ang = ((jet_p4.eta - jet_p4.sum().eta) ** 2 + (jet_p4.phi - jet_p4.sum().phi)) ** .5
        return np.sum( z ** k * ((ang / R) ** beta) )

    def _2ecf(self, jet_p4, beta):
        z = jet_p4.pt / np.sum(jet_p4.pt)
        ecf = 0.0
        for i in range(len(z)):
            for j in range(len(z)):
                if i < j:
                    ang_ij = ((jet_p4[i].eta - jet_p4.[j].eta) ** 2 + (jet_p4[i].phi - jet_p4[j].phi)) ** .5
                    ecf += z[i] * z[j] * ang_ij ** beta
        return ecf 
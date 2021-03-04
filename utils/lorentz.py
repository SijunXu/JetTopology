import numpy as np
import awkward
import uproot_methods

def get_p4(jet_particle, zero_mass=True, singleJet=False):
    ### return 4-p of particles of a set of jets
    ### (nb_jets, max_par, (pt, eta, phi, mass)) 
    jet_particle = jet_particle.astype(np.float32)
    if singleJet:
        mask = jet_particle[:, 0] > 0
        n_particles = np.sum(mask)
    else:
        mask = jet_particle[:, :, 0] > 0
        n_particles = np.sum(mask, axis=1)
    if singleJet:
        pt = jet_particle[mask][0]
        eta = jet_particle[mask][1]
        phi = jet_particle[mask][2]
        if zero_mass:
            mass = np.zeros(jet_particle[mask][0].shape) 
        else:
            mass = jet_particle[mask][3]
    else:
        pt = awkward.JaggedArray.fromcounts(n_particles, jet_particle[mask][:, 0])
        eta = awkward.JaggedArray.fromcounts(n_particles, jet_particle[mask][:, 1])
        phi = awkward.JaggedArray.fromcounts(n_particles, jet_particle[mask][:, 2])
        if zero_mass:
            mass = awkward.JaggedArray.fromcounts(n_particles, np.zeros(jet_particle[mask][:, 0].shape))
        else:
            mass = awkward.JaggedArray.fromcounts(n_particles, jet_particle[mask][:, 3])    
    p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(pt, eta, phi, mass)
    return p4
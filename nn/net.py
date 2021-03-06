import torch
import torch.nn as nn
import torch.nn.functional as F


class fcn_net(nn.Module):
    '''
    nn model for tabuler data
    shape of output: (n_data, 1)
    '''    
    def __init__(self, layers=[128, 128, 128, 64, 16], indim=50, BN=True, net_classify=True):
        super(fcn_net, self).__init__()
        def com_layer(d_in, d_out, BN=False):
            if BN:
                return nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(inplace=True), nn.BatchNorm1d(d_out))
            else:
                return nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(inplace=True))
            
        self.layers = nn.ModuleList([
            com_layer(indim, layers[i], BN) if i==0
            else com_layer(layers[i-1], layers[i], BN) for i in range(len(layers))
                                    ])
        self.net_classify = net_classify
        if self.net_classify:
            self.classify = nn.Sequential(nn.Linear(layers[-1], 1), nn.Sigmoid())
        
    def forward(self, x):
        for l in self.layers:
             x = l(x)
        if self.net_classify:
            x = self.classify(x)
        return x


class TopoObsNet(nn.Module):
    '''
    input: topological features + tau1-6
    '''
    def __init__(
        self, 
        topo_layers=[128, 128, 128, 64, 16], 
        topo_indim=50, 
        obs_layers=[128, 128, 16], 
        obs_indim=6,
        show_mid=False
        ):
        super(TopoObsNet, self).__init__()
        self.topo_net = fcn_net(layers=topo_layers, indim=topo_indim)
        self.obs_net = fcn_net(layers=obs_layers, indim=obs_indim)
        self.classify = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        self.show_mid = show_mid

    def forward(self, x, obs):        
        topo_out = self.topo_net(x)
        obs_out = self.obs_net(obs)
        out = torch.cat((topo_out.view(-1, 1), obs_out.view(-1, 1)), dim=1)
        out = self.classify(out)
        if self.show_mid:
            return out, topo_out, obs_out
        else:
            return out


class DeepSet(nn.Module):
    '''
    DeepSet NN transforms data as :
    out = rho( sum( weight * phi(x), dim=1 ) )
    '''
    def __init__(self, indim, phi_layers, rho_layers, BN=True):
        super(DeepSet, self).__init__()
        self.phi = fcn_net(layers=phi_layers, indim=indim, BN=False, net_classify=False)
        self.rho = fcn_net(layers=rho_layers, indim=phi_layers[-1], BN=BN, net_classify=False)

    def forward(self, x, weight):
        x = (weight * self.phi(x)).sum(dim=1)
        out = self.rho(x)
        return out

class PersNet(nn.Module):
    '''
    input b0 and b1 features, each kind of feature under the transformation of DeepSet, then concatenate for classification 
    '''
    def __init__(self, b0_dim, b1_dim, b0_phi_layers, b1_phi_layers, b0_rho_layers, b1_rho_layers, fc_layers, BN=True):
        super(PersNet, self).__init__()

        self.b0_net = DeepSet(indim=b0_dim, phi_layers=b0_phi_layers, rho_layers=b0_rho_layers, BN=BN)
        self.b1_net = DeepSet(indim=b1_dim, phi_layers=b1_phi_layers, rho_layers=b1_rho_layers, BN=BN)

        self.fc = fcn_net(layers=fc_layers, indim=(b0_rho_layers[-1] + b1_rho_layers[-1]), BN=BN, net_classify=False)
        self.classify = nn.Sequential(nn.Linear(fc_layers[-1], 1), nn.Sigmoid())
    
    def forward(self, b0, b0_weight, b1, b1_weight):
        b0_feat = self.b0_net(b0, b0_weight)
        b1_feat = self.b1_net(b1, b1_weight)
        x = torch.cat([b0_feat, b1_feat], dim=1)
        out = self.classify(self.fc(x))
        return out

class PersNetkNN(nn.Module):
    '''
    use information of b0 for classification 
    '''
    def __init__(self, b0_dim, b0_phi_layers, b0_rho_layers, fc_layers, BN=True):
        super(PersNetkNN, self).__init__()

        self.b0_net = DeepSet(indim=b0_dim, phi_layers=b0_phi_layers, rho_layers=b0_rho_layers, BN=BN)
        self.fc = fcn_net(layers=fc_layers, indim=b0_rho_layers[-1], BN=BN, net_classify=False)
        self.classify = nn.Sequential(nn.Linear(fc_layers[-1], 1), nn.Sigmoid())

    def forward(self, b0, b0_weight):
        b0_feat = self.b0_net(b0, b0_weight)                
        out = self.classify(self.fc(b0_feat))
        return out
        
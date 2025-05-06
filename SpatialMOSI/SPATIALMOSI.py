import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch_scatter
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from .GAT_conv import GATConv
import torch.nn.init as init
from torch.nn.parameter import Parameter

class Spatial_MSI(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(Spatial_MSI, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.dec1 = torch.nn.Linear(out_dim, num_hidden)
        self.dec2 = torch.nn.Linear(num_hidden, in_dim)
    def forward(self, features, edge_index, edge_CSL):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        h1_neg = F.elu(self.conv1(features, edge_CSL))
        h2_neg = self.conv2(h1_neg, edge_CSL, attention=False)
        
        rec = self.dec1(h2)
        rec = F.elu(rec)
        rec = self.dec2(rec)
        h_pos = self.CSL(h2, edge_index)
        return h2, h_pos, h2_neg, rec
    def CSL(self, h, edge_index):
        node_features = h.index_select(0, edge_index[1])
        h_agg = torch_scatter.scatter_mean(node_features, edge_index[0], dim=0)
        return h_agg

class AttentionLayer(nn.Module):
    
    def __init__(self, in_feat, out_feat):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb_omic1, emb_omic2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb_omic1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb_omic2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha 
    
class Spatial_MOSI_att(nn.Module):
    def __init__(self, hidden_dims_1, hidden_dims_2):
        super(Spatial_MOSI_att, self).__init__()
        [in_dim_1, num_hidden_1, out_dim_1] = hidden_dims_1
        [in_dim_2, num_hidden_2, out_dim_2] = hidden_dims_2
        self.conv1_mod1 = GATConv(in_dim_1, num_hidden_1, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2_mod1 = GATConv(num_hidden_1, out_dim_1, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv1_mod2 = GATConv(in_dim_2, num_hidden_2, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2_mod2 = GATConv(num_hidden_2, out_dim_2, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.atten_cross = AttentionLayer(out_dim_1, out_dim_2)
        self.dec1_mod1 = torch.nn.Linear(out_dim_1, num_hidden_1)
        self.dec2_mod1 = torch.nn.Linear(num_hidden_1, in_dim_1)
        self.dec1_mod2 = torch.nn.Linear(out_dim_2, num_hidden_2)
        self.dec2_mod2 = torch.nn.Linear(num_hidden_2, in_dim_2)

    def forward(self, features_1, features_2, edge_index_1, edge_index_2, edge_CSL):
        h1_mod1 = F.elu(self.conv1_mod1(features_1, edge_index_1))
        h2_mod1 = self.conv2_mod1(h1_mod1, edge_index_1, attention=False)
        h1_mod2 = F.elu(self.conv1_mod2(features_2, edge_index_2))
        h2_mod2 = self.conv2_mod2(h1_mod2, edge_index_2, attention=False)

        emb, atts = self.atten_cross(h2_mod1, h2_mod2)
        rec_mod1 = self.dec1_mod1(emb)
        rec_mod1 = F.elu(rec_mod1)
        rec_mod1 = self.dec2_mod1(rec_mod1)
        rec_mod2 = self.dec1_mod2(emb)
        rec_mod2 = F.elu(rec_mod2)
        rec_mod2 = self.dec2_mod2(rec_mod2)
        h1_mod1_neg = F.elu(self.conv1_mod1(features_1, edge_CSL))
        h2_mod1_neg = self.conv2_mod1(h1_mod1_neg, edge_CSL, attention=False)
        h1_mod2_neg = F.elu(self.conv1_mod2(features_2, edge_CSL))
        h2_mod2_neg = self.conv2_mod2(h1_mod2_neg, edge_CSL, attention=False)
        #emb_neg, alpha_neg = self.atten_cross(h2_mod1_neg, h2_mod2_neg)
        h_mod1_pos = self.CSL(h2_mod1, edge_index_1)
        h_mod2_pos = self.CSL(h2_mod2, edge_index_2)
        h2_csl1_neg = self.CSL(h2_mod1, edge_CSL)
        h2_csl2_neg = self.CSL(h2_mod2, edge_CSL)#h2_csl1_neg, h2_csl2_neg,
        return h2_mod1, h2_mod2, h_mod1_pos, h_mod2_pos, h2_mod1_neg, h2_mod2_neg,  emb, rec_mod1, rec_mod2,atts
    def CSL(self, h, edge_index):
        node_features = h.index_select(0, edge_index[1])
        h_agg = torch_scatter.scatter_mean(node_features, edge_index[0], dim=0)
        return h_agg

class AttentionLayer_triplet(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(AttentionLayer_triplet, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb_omic1, emb_omic2, emb_omic3):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb_omic1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb_omic2), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb_omic3), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha     

class Spatial_MOSI_triple(nn.Module):
    def __init__(self, hidden_dims_1, hidden_dims_2, hidden_dims_3):
        super(Spatial_MOSI_triple, self).__init__()
        [in_dim_1, num_hidden_1, out_dim_1] = hidden_dims_1
        [in_dim_2, num_hidden_2, out_dim_2] = hidden_dims_2
        [in_dim_3, num_hidden_3, out_dim_3] = hidden_dims_3
        self.conv1_mod1 = GATConv(in_dim_1, num_hidden_1, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2_mod1 = GATConv(num_hidden_1, out_dim_1, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv1_mod2 = GATConv(in_dim_2, num_hidden_2, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2_mod2 = GATConv(num_hidden_2, out_dim_2, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv1_mod3 = GATConv(in_dim_3, num_hidden_3, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2_mod3 = GATConv(num_hidden_3, out_dim_3, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.atten_cross = AttentionLayer_triplet(out_dim_1, out_dim_2)
        self.dec1_mod1 = torch.nn.Linear(out_dim_1, num_hidden_1)
        self.dec2_mod1 = torch.nn.Linear(num_hidden_1, in_dim_1)
        self.dec1_mod2 = torch.nn.Linear(out_dim_2, num_hidden_2)
        self.dec2_mod2 = torch.nn.Linear(num_hidden_2, in_dim_2)
        self.dec1_mod3 = torch.nn.Linear(out_dim_3, num_hidden_3)
        self.dec2_mod3 = torch.nn.Linear(num_hidden_3, in_dim_3)

    def forward(self, features_1, features_2, features_3, edge_index_1, edge_index_2, edge_index_3,edge_CSL):
        h1_mod1 = F.elu(self.conv1_mod1(features_1, edge_index_1))
        h2_mod1 = self.conv2_mod1(h1_mod1, edge_index_1, attention=False)
        h1_mod2 = F.elu(self.conv1_mod2(features_2, edge_index_2))
        h2_mod2 = self.conv2_mod2(h1_mod2, edge_index_2, attention=False)
        h1_mod3 = F.elu(self.conv1_mod3(features_3, edge_index_3))
        h2_mod3 = self.conv2_mod3(h1_mod3, edge_index_3, attention=False)
        
        emb, alpha = self.atten_cross( h2_mod1, h2_mod2, h2_mod3)
        rec_mod1 = self.dec1_mod1(emb)
        rec_mod1 = F.elu(rec_mod1)
        rec_mod1 = self.dec2_mod1(rec_mod1)
        rec_mod2 = self.dec1_mod2(emb)
        rec_mod2 = F.elu(rec_mod2)
        rec_mod2 = self.dec2_mod2(rec_mod2)
        rec_mod3 = self.dec1_mod3(emb)
        rec_mod3 = F.elu(rec_mod3)
        rec_mod3 = self.dec2_mod3(rec_mod3)
        h1_mod1_neg = F.elu(self.conv1_mod1(features_1, edge_CSL))
        h2_mod1_neg = self.conv2_mod1(h1_mod1_neg, edge_CSL, attention=False)
        h1_mod2_neg = F.elu(self.conv1_mod2(features_2, edge_CSL))
        h2_mod2_neg = self.conv2_mod2(h1_mod2_neg, edge_CSL, attention=False)
        h1_mod3_neg = F.elu(self.conv1_mod3(features_3, edge_CSL))
        h2_mod3_neg = self.conv2_mod3(h1_mod3_neg, edge_CSL, attention=False)
        h_mod1_pos = self.CSL(h2_mod1, edge_index_1)
        h_mod2_pos = self.CSL(h2_mod2, edge_index_2)
        h_mod3_pos = self.CSL(h2_mod3, edge_index_3)
        return h2_mod1, h2_mod2, h2_mod3, h_mod1_pos, h_mod2_pos, h_mod3_pos, h2_mod1_neg, h2_mod2_neg, h2_mod3_neg, emb, rec_mod1, rec_mod2, rec_mod3
    def CSL(self, h, edge_index):
        node_features = h.index_select(0, edge_index[1])
        h_agg = torch_scatter.scatter_mean(node_features, edge_index[0], dim=0)
        return h_agg
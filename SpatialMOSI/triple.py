import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from .SPATIALMOSI import Spatial_MOSI_triple
from torch import nn
import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
def train_SpatialMOSI_triple(adata_1,adata_2, adata_3, chr=False, beta=0.5, alpha=1,
            hidden_dims_1=[512, 30], hidden_dims_2=[40, 30], hidden_dims_3=[60,30], n_epochs=1000, lr=0.001, key_added='embedding',
            gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
            random_seed=666,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


    edgeList_1 = adata_1.uns['adj']
    edgeList_2 = adata_2.uns['adj']
    edgeList_3 = adata_3.uns['adj']

    graphcsl = adata_1.uns['edgecsl']
    x_value1 = adata_1.X
    x_value2 = adata_2.X
    x_value3 = adata_3.X
    if hasattr(x_value1, 'todense'):
        x_value1=x_value1.todense()
    else:
        x_value1=np.array(x_value1)
    if hasattr(x_value2, 'todense'):
        x_value2=x_value2.todense()
    else:
        x_value2=np.array(x_value2)
    if chr:
        x_value3 = adata_3.obsm['X_lsi']
        if hasattr(x_value3, 'todense'):
            x_value3 = x_value3.todense()
        else:
            x_value3 = np.array(x_value3)
    else:
        x_value3 = adata_3.X
        if hasattr(x_value3, 'todense'):
            x_value3 = x_value3.todense()
        else:
            x_value3 = np.array(x_value3)

    data_1 = Data(edge_index=torch.LongTensor(np.array([edgeList_1[0], edgeList_1[1]])),
                edge_CSL = torch.LongTensor(np.array([graphcsl[0], graphcsl[1]])),
                prune_edge_index=torch.LongTensor(np.array([])),
                x=torch.FloatTensor(x_value1))
    data_1 = data_1.to(device)
    data_2 = Data(edge_index=torch.LongTensor(np.array([edgeList_2[0], edgeList_2[1]])),
                edge_CSL = torch.LongTensor(np.array([graphcsl[0], graphcsl[1]])),
                prune_edge_index=torch.LongTensor(np.array([])),
                x=torch.FloatTensor(x_value2))
    data_2 = data_2.to(device)
    data_3 = Data(edge_index=torch.LongTensor(np.array([edgeList_3[0], edgeList_3[1]])),
                edge_CSL = torch.LongTensor(np.array([graphcsl[0], graphcsl[1]])),
                prune_edge_index=torch.LongTensor(np.array([])),
                x=torch.FloatTensor(x_value3))
    data_3 = data_3.to(device)
    model = Spatial_MOSI_triple(hidden_dims_1=[data_1.x.shape[1], hidden_dims_1[0], hidden_dims_1[1]], hidden_dims_2=[data_2.x.shape[1], hidden_dims_2[0], hidden_dims_2[1]], hidden_dims_3=[data_3.x.shape[1], hidden_dims_3[0], hidden_dims_3[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)
    loss_mod_values = []
    loss_mse_values = []
    loss_csl_values = []
    loss_values = []
    for epoch in tqdm(range(0,n_epochs)):
        model.train()
        optimizer.zero_grad()
        emb_latent_omics1, emb_latent_omics2, emb_latent_omics3, h_mod1_pos, h_mod2_pos, h_mod3_pos, h2_mod1_neg, h2_mod2_neg, h2_mod3_neg, emb, rec_1, rec_2, rec_3=model(data_1.x, data_2.x, data_3.x, data_1.edge_index, data_2.edge_index, data_3.edge_index, data_1.edge_CSL)

        rec_loss1 = F.mse_loss(data_1.x, rec_1)
        rec_loss2 = F.mse_loss(data_2.x, rec_2)
        rec_loss3 = F.mse_loss(data_3.x, rec_3)
        rec_loss = rec_loss1 + rec_loss2 + rec_loss3
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        loss_csl = triplet_loss(emb_latent_omics1, h_mod1_pos, h2_mod1_neg) + triplet_loss(emb_latent_omics2, h_mod2_pos, h2_mod2_neg) + triplet_loss(emb_latent_omics3, h_mod3_pos, h2_mod3_neg)
        loss_mod1 = triplet_loss(emb_latent_omics1, emb_latent_omics2, h2_mod1_neg) + triplet_loss(emb_latent_omics1, emb_latent_omics3, h2_mod1_neg)
        loss_mod2 = triplet_loss(emb_latent_omics2, emb_latent_omics1, h2_mod2_neg) + triplet_loss(emb_latent_omics2, emb_latent_omics3, h2_mod2_neg)
        loss_mod3 = triplet_loss(emb_latent_omics3, emb_latent_omics1, h2_mod3_neg) + triplet_loss(emb_latent_omics3, emb_latent_omics2, h2_mod3_neg)
        loss_mod = loss_mod1 + loss_mod2 + loss_mod3
        loss = rec_loss  + alpha*loss_mod + beta*loss_csl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    with torch.no_grad():
        emb_latent_omics1, emb_latent_omics2, emb_latent_omics3, h_mod1_pos, h_mod2_pos, h_mod3_pos, h2_mod1_neg, h2_mod2_neg, h2_mod3_neg, emb, rec_1, rec_2, rec_3=model(data_1.x, data_2.x, data_3.x, data_1.edge_index, data_2.edge_index, data_3.edge_index,data_1.edge_CSL)
    

    adata_1.obsm['omic'] = emb_latent_omics1.cpu().detach().numpy()
    adata_2.obsm['omic'] = emb_latent_omics2.cpu().detach().numpy()
    adata_3.obsm['omic'] = emb_latent_omics3.cpu().detach().numpy()
    adata_1.obsm['omics'] = emb.cpu().detach().numpy()


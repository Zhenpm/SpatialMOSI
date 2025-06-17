import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from .utils import generate_csl_graph,create_dictionary_mnn
from .SPATIALMOSI import Spatial_MOSI_att, Spatial_MOSI_triple, Spatial_MSI
from torch import nn
import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#import matplotlib.pyplot as plt
def train_SpatialMSI(adata,  chr = True, beta=0.5, lamda=1,
                    hidden_dims=[512, 30], alpha=1,pre_epochs=500, n_epochs=1000, lr=0.001, key_added='embedding',
                    gradient_clipping=5., weight_decay=0.0001, margin=1.0, show=False,
                    random_seed=666, csp_groups=None, k_csps=50, save_embedding=None,save_model=None,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids = np.array(adata.obs['batch_name'].unique())
    edgeList = adata.uns['adj']
    graphcsl = adata.uns['edgecsl']
    if chr:
        x_value = adata.obsm['X_lsi']
        if hasattr(x_value, 'todense'):
            x_value = x_value.todense()
        else:
            x_value = np.array(x_value)
    else:
        x_value = adata.X
        if hasattr(x_value, 'todense'):
            x_value = x_value.todense()
        else:
            x_value = np.array(x_value)
    data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                edge_CSL = torch.LongTensor(np.array([graphcsl[0], graphcsl[1]])),
                x=torch.FloatTensor(x_value))
    data = data.to(device)
    model = Spatial_MSI(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if show:
        print(model)

    print('Pretraining')
    for epoch in tqdm(range(0, pre_epochs)):
        model.train()
        optimizer.zero_grad()
        z, z_pos, z_neg,  rec = model(data.x, data.edge_index, data.edge_CSL)
        rec_loss = F.mse_loss(data.x, rec)
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        loss_csl = triplet_loss(z, z_pos, z_neg) 
        loss = alpha*rec_loss + beta * loss_csl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

    with torch.no_grad():
        z, _,_,_ = model(data.x, data.edge_index, data.edge_CSL)
    adata.obsm['omic'] = z.cpu().detach().numpy()
    
    print('Training for multi slices...')
    for epoch in tqdm(range(pre_epochs, n_epochs)):
        if epoch % 100 == 0 or epoch == 500:
            if show:
                print('Update spot triplets at epoch ' + str(epoch))
            adata.obsm['omic'] = z.cpu().detach().numpy()

            csp_dict = create_dictionary_mnn(adata, use_rep='omic', batch_name='batch_name', k=k_csps,
                                                       csp_groups=csp_groups)

            anchor_ind = []
            positive_ind = []
            negative_ind = []
            for csps in csp_dict.keys():  
                batchname_list = adata.obs['batch_name'][csp_dict[csps].keys()]
                cell_batch_dict = dict()
                for batch_id in range(len(section_ids)):
                    cell_batch_dict[section_ids[batch_id]] = adata.obs_names[
                        adata.obs['batch_name'] == section_ids[batch_id]].values

                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in csp_dict[csps].keys():
                    anchor_list.append(anchor)
                    pos_cell = csp_dict[csps][anchor][0]
                    positive_list.append(pos_cell)
                    slice_size = len(cell_batch_dict[batchname_list[anchor]])
                    negative_list.append(
                        cell_batch_dict[batchname_list[anchor]][np.random.randint(slice_size)])

                batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

        model.train()
        optimizer.zero_grad()
        z,z_pos, z_neg, rec = model(data.x, data.edge_index, data.edge_CSL)

        rec_loss = F.mse_loss(data.x, rec)
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        loss_csl = triplet_loss(z, z_pos, z_neg) 
        anchor_arr= z[anchor_ind,]
        positive_arr = z[positive_ind,]
        negative_arr = z[negative_ind,]
        loss_slice = triplet_loss(anchor_arr, positive_arr, negative_arr)
        loss = alpha*rec_loss + lamda*loss_slice + beta*loss_csl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    model.eval()
    adata.obsm[key_added] = z.cpu().detach().numpy()
    adata.obsm['negemb'] = z_neg.cpu().detach().numpy()
    if (isinstance(save_embedding, str)):
        np.save(save_embedding, adata.obsm[key_added], allow_pickle=True, fix_imports=True)
        print('Successfully save final embedding at {}.'.format(save_embedding)) if (show) else None

    if (isinstance(save_model, str)):
        torch.save(model, save_model)
        print('Successfully export final model at {}.'.format(save_model)) if (show) else None

def train_SpatialMOSI(adata_1, adata_2, chr=False, alpha=1, beta = 0.5, gamma=1, lamda=1, omics2_csps=True,
                    hidden_dims_1=[512, 30],hidden_dims_2=[40, 30], n_epochs=1000, lr=0.001, key_added='embedding',
                    gradient_clipping=5., weight_decay=0.0001, margin=1.0, show=False, key_added_combined='embedding_omics',
                    random_seed=666, csp_groups=None, k_csps=100, pre_epochs = 500, save_embedding_final=None,save_model=None,
                    save_embedding_omic1=None, save_embedding_omic2=None,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids_1 = np.array(adata_1.obs['batch_name'].unique())
    section_ids_2 = np.array(adata_2.obs['batch_name'].unique())
    edgeList_1 = adata_1.uns['adj']
    edgeList_2 = adata_2.uns['adj']
    
    graphcsl = adata_1.uns['edgecsl']
    x_value = adata_1.X
    if hasattr(x_value, 'todense'):
        x_value = x_value.todense()
    else:
        x_value = np.array(x_value)
    if chr:
        x_value2 = adata_2.obsm['X_lsi']
        if hasattr(x_value2, 'todense'):
            x_value2 = x_value2.todense()
        else:
            x_value2 = np.array(x_value2)
    else:
        x_value2 = adata_2.X
        if hasattr(x_value2, 'todense'):
            x_value2 = x_value2.todense()
        else:
            x_value2 = np.array(x_value2)
    data_1 = Data(edge_index=torch.LongTensor(np.array([edgeList_1[0], edgeList_1[1]])),
                edge_CSL = torch.LongTensor(np.array([graphcsl[0], graphcsl[1]])),
                x=torch.FloatTensor(x_value))
    data_1 = data_1.to(device)
    data_2 = Data(edge_index=torch.LongTensor(np.array([edgeList_2[0], edgeList_2[1]])),
                edge_CSL = torch.LongTensor(np.array([graphcsl[0], graphcsl[1]])),
                x=torch.FloatTensor(x_value2))
    data_2 = data_2.to(device)
    model = Spatial_MOSI_att(hidden_dims_1=[data_1.x.shape[1], hidden_dims_1[0], hidden_dims_1[1]], hidden_dims_2=[data_2.x.shape[1], hidden_dims_2[0], hidden_dims_2[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if show:
        print(model)
    print('Pretraining...')
    for epoch in tqdm(range(0, pre_epochs)):
        model.train()
        optimizer.zero_grad()
        z_1, z_2, z_1_pos, z_2_pos, z_1_neg, z_2_neg, z, rec_1, rec_2, atts = model(data_1.x, data_2.x, data_1.edge_index, data_2.edge_index, data_1.edge_CSL)
        rec_loss2 = F.mse_loss(data_2.x, rec_2)
        rec_loss1 = F.mse_loss(data_1.x, rec_1)
        rec_loss = rec_loss1 + rec_loss2
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        loss_mod = triplet_loss(z_1, z_2, z_1_neg) + triplet_loss(z_2, z_1, z_2_neg)
        loss_csl = triplet_loss(z_1, z_1_pos, z_1_neg) + triplet_loss(z_2, z_2_pos, z_2_neg)
        loss = alpha*rec_loss + gamma*loss_mod + beta*loss_csl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    with torch.no_grad():
        z_1, z_2,_,_,_,_,z,rec_1, rec_2, atts = model(data_1.x, data_2.x, data_1.edge_index, data_2.edge_index, data_1.edge_CSL)

    adata_1.obsm['omic'] = z_1.cpu().detach().numpy()
    adata_2.obsm['omic'] = z_2.cpu().detach().numpy()
    adata_1.obsm['omics'] = z.cpu().detach().numpy()
    

    print('Training for multi slices...')
    for epoch in tqdm(range(pre_epochs, n_epochs)):
        if epoch % 100 == 0 or epoch == 500:
            if show:
                print('Update spot triplets at epoch ' + str(epoch))
            adata_1.obsm['omic'] = z_1.cpu().detach().numpy()
            adata_2.obsm['omic'] = z_2.cpu().detach().numpy()
            csps_dict_1 = create_dictionary_mnn(adata_1, use_rep='omic', batch_name='batch_name', k=k_csps,
                                                       iter_comb =csp_groups)

            anchor_ind_1 = []
            positive_ind_1 = []
            negative_ind_1 = []
            for batch_pair in csps_dict_1.keys():  # pairwise compare for multiple batches
                batchname_list = adata_1.obs['batch_name'][csps_dict_1[batch_pair].keys()]
                #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
                #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

                cellname_by_batch_dict = dict()
                for batch_id in range(len(section_ids_1)):
                    cellname_by_batch_dict[section_ids_1[batch_id]] = adata_1.obs_names[
                        adata_1.obs['batch_name'] == section_ids_1[batch_id]].values

                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in csps_dict_1[batch_pair].keys():
                    anchor_list.append(anchor)
                    ## np.random.choice(mnn_dict[batch_pair][anchor])
                    positive_spot = csps_dict_1[batch_pair][anchor][0]  # select the first positive spot
                    positive_list.append(positive_spot)
                    section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                    negative_list.append(
                        cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                batch_as_dict = dict(zip(list(adata_1.obs_names), range(0, adata_1.shape[0])))
                anchor_ind_1 = np.append(anchor_ind_1, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind_1 = np.append(positive_ind_1, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind_1 = np.append(negative_ind_1, list(map(lambda _: batch_as_dict[_], negative_list)))
            if(omics2_csps):
                csps_dict_2 = create_dictionary_mnn(adata_2, use_rep='omic', batch_name='batch_name', k=k_csps,
                                                        iter_comb =csp_groups)
                anchor_ind_2 = []
                positive_ind_2 = []
                negative_ind_2 = []
                for batch_pair in csps_dict_2.keys():  # pairwise compare for multiple batches
                    batchname_list = adata_2.obs['batch_name'][csps_dict_2[batch_pair].keys()]
                    #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
                    #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(section_ids_2)):
                        cellname_by_batch_dict[section_ids_2[batch_id]] = adata_2.obs_names[
                            adata_2.obs['batch_name'] == section_ids_2[batch_id]].values

                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for anchor in csps_dict_2[batch_pair].keys():
                        anchor_list.append(anchor)
                        ## np.random.choice(mnn_dict[batch_pair][anchor])
                        positive_spot = csps_dict_2[batch_pair][anchor][0]  # select the first positive spot
                        positive_list.append(positive_spot)
                        section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                        negative_list.append(
                            cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                    batch_as_dict = dict(zip(list(adata_2.obs_names), range(0, adata_2.shape[0])))
                    anchor_ind_2 = np.append(anchor_ind_2, list(map(lambda _: batch_as_dict[_], anchor_list)))
                    positive_ind_2 = np.append(positive_ind_2, list(map(lambda _: batch_as_dict[_], positive_list)))
                    negative_ind_2 = np.append(negative_ind_2, list(map(lambda _: batch_as_dict[_], negative_list)))

        model.train()
        optimizer.zero_grad()
        z_1, z_2, z_1_pos, z_2_pos, z_1_neg, z_2_neg, z, rec_1, rec_2, atts = model(data_1.x, data_2.x, data_1.edge_index, data_2.edge_index, data_1.edge_CSL)
        rec_loss2 = F.mse_loss(data_2.x, rec_2)
        rec_loss1 = F.mse_loss(data_1.x, rec_1)
        rec_loss = rec_loss1+ rec_loss2
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        loss_mod = triplet_loss(z_1, z_2, z_1_neg) + triplet_loss(z_2, z_1, z_2_neg)
        loss_csl = triplet_loss(z_1, z_1_pos, z_1_neg) + triplet_loss(z_2, z_2_pos, z_2_neg)
        anchor_arr_1 = z_1[anchor_ind_1,]
        positive_arr_1 = z_1[positive_ind_1,]
        negative_arr_1 = z_1[negative_ind_1,]
        if (omics2_csps):
            anchor_arr_2 = z_2[anchor_ind_2,]
            positive_arr_2 = z_2[positive_ind_2,]
            negative_arr_2 = z_2[negative_ind_2,]
        else:
            anchor_arr_2 = z_2[anchor_ind_1,]
            positive_arr_2 = z_2[positive_ind_1,]
            negative_arr_2 = z_2[negative_ind_1,]


        tri_output_1 = triplet_loss(anchor_arr_1, positive_arr_1, negative_arr_1)
        tri_output_2 = triplet_loss(anchor_arr_2, positive_arr_2, negative_arr_2)
        loss_slice = tri_output_1 + tri_output_2
        loss = alpha*rec_loss + gamma*loss_mod + beta*loss_csl + lamda * loss_slice 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    model.eval()
    adata_1.obsm[key_added] = z_1.cpu().detach().numpy()
    adata_2.obsm[key_added] = z_2.cpu().detach().numpy()
    adata_1.obsm[key_added_combined] = z.cpu().detach().numpy()
    adata_1.obsm['rec'] = rec_1.cpu().detach().numpy()
    adata_2.obsm['rec'] = rec_2.cpu().detach().numpy()
    adata_1.obsm['att'] = atts.cpu().detach().numpy()

    if (isinstance(save_embedding_final, str)):
        np.save(save_embedding_final, adata_1.obsm[key_added_combined], allow_pickle=True, fix_imports=True)
        print('Successfully save final embedding at {}.'.format(save_embedding_final)) if (show) else None

    if (isinstance(save_embedding_omic1, str)):
        np.save(save_embedding_omic1, adata_1.obsm[key_added], allow_pickle=True, fix_imports=True)
        print('Successfully save omic1 embedding at {}.'.format(save_embedding_omic1)) if (show) else None

    if (isinstance(save_embedding_omic2, str)):
        np.save(save_embedding_omic2, adata_1.obsm[key_added], allow_pickle=True, fix_imports=True)
        print('Successfully save omic2 embedding at {}.'.format(save_embedding_omic2)) if (show) else None

    if (isinstance(save_model, str)):
        torch.save(model, save_model)
        print('Successfully export final model at {}.'.format(save_model)) if (show) else None

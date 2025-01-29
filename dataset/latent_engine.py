import os
import json
import numpy as np
import pandas as pd
import torch
from vae.model import Decoder
from vae.utils import load_dataset
import torch.nn.functional as F


def get_input_train(args):
    dataname = args.dataname
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    weight_dir = f'{curr_dir}/weight/{dataname}/'
    embedding_save_path = f'{curr_dir}/vae/weight/{dataname}/latent.npy'
    latent = torch.tensor(np.load(embedding_save_path)).float()

    return latent, curr_dir, dataset_dir, weight_dir, info


def get_input_generate(args):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'
    weight_dir = f'{curr_dir}/weight/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    data = load_dataset(dataname)
    x_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1
    task_type = info['task_type']
    weight_dir = f'{curr_dir}/weight/{dataname}'
    embedding_save_path = f'{curr_dir}/vae/weight/{dataname}/latent.npy'
    latent = torch.tensor(np.load(embedding_save_path)).float()

    pre_decoder = Decoder(input_dim=data.x.shape[1] + data.y.max().item() + 1, latent_dim=128, hidden_dim=512)
    decoder_save_path = f'{curr_dir}/vae/weight/{dataname}/decoder.pt'
    pre_decoder.load_state_dict(torch.load(decoder_save_path))

    info['pre_decoder'] = pre_decoder
    info['x_dim'] = x_dim
    info['num_classes'] = num_classes

    return latent, curr_dir, dataset_dir, weight_dir, info


@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device):
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    pre_decoder = info['pre_decoder']

    syn_data = syn_data.reshape(syn_data.shape[0], -1)
    norm_input = pre_decoder(torch.tensor(syn_data))
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim = -1))

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target


@torch.no_grad()
def get_ori_data(syn_data, info, device, threshold):
    pre_decoder = info['pre_decoder']
    x_dim = info['x_dim']
    num_classes = info['num_classes']

    norm_input = pre_decoder(torch.tensor(syn_data))

    recon_x = norm_input[:, :x_dim]           
    recon_y = norm_input[:, x_dim:]           

    recon_x = torch.sigmoid(recon_x)           
    recon_y = F.softmax(recon_y, dim=1)           

    threshold = threshold
    binary_recon_x = (recon_x >= threshold).float()

    pred_labels = recon_y.argmax(dim=1)

    syn_final_data = torch.cat([binary_recon_x, pred_labels.unsqueeze(1).float()], dim=1)

    return syn_final_data


def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df
    

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat


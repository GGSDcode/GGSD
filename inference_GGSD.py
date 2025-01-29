import os
import numpy as np
import torch
import argparse
import warnings
import time
import pandas as pd
from tqdm import tqdm
from dataset.latent_engine import get_input_generate, recover_data, split_num_cat_target, get_ori_data
from utils.diffusion import sample
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from ggsd.gnn_denoiser import Model, GGSD_Diffusion
from partition.SEPG.utils import PartitionTree
from partition.SEPN.prepare_nodeData import update_node
from partition.utils import extract_cluster_assignment, get_all_partitions
from utils.config import get_args
from utils.tools import create_logger


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path
    guidance_scale = args.guidance_scale

    logger = create_logger(args.log_dir)
    logger.info(f'logging directory created at {args.log_dir}')

    latent, _, _, weight_path, info = get_input_generate(args)
    edge_index = torch.tensor(info['edge_index'], dtype=torch.long)
    logger.info(f'in sampling process, edge_index shape is: {edge_index.shape}')

    in_dim = latent.shape[1] 
    mean = latent.mean(0)
    gnn_denoiser = GGSD_Diffusion(in_dim, 1024).to(device)
    
    model = Model(gnn_denoiser = gnn_denoiser, hid_dim = latent.shape[1]).to(device)
    model.load_state_dict(torch.load(f'{weight_path}/{args.method}.pt'))
    
    start_time = time.time()
    num_samples = latent.shape[0]
    sample_dim = in_dim

    tree_depth = args.tree_depth
    SE_k = args.SE_k
    
    num_nodes = latent.shape[0]
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj_matrix[edge_index[0], edge_index[1]] = 1

    undirected_adj = np.array(adj_matrix)
    y = PartitionTree(adj_matrix=undirected_adj)
    y.build_coding_tree(tree_depth)
    T = update_node(y.tree_node)
    S_edge_index_list = extract_cluster_assignment(T, tree_depth=tree_depth)
    S_edge_index_list = [torch.LongTensor(S_edge_index_list[i]).T for i in range(len(S_edge_index_list))]

    partitions_per_layer = get_all_partitions(S_edge_index_list)
    for i in range(len(partitions_per_layer)):
        point_list = []
        for j in partitions_per_layer[i].keys():
            point_list.extend(partitions_per_layer[i][j])

    partitions_k = partitions_per_layer[SE_k]

    latents = torch.randn([num_samples, sample_dim])
    x_next_full = torch.zeros_like(latents).to(device)

    graph_batch_nodes_set = set()
    for i, subgraph_idx in enumerate(tqdm(partitions_k.keys(), total=len(partitions_k.keys()), desc="Generating")):
        nodes_tensor = torch.tensor(partitions_k[subgraph_idx], dtype=torch.long)
        inputs = latents[nodes_tensor]
        subgraph_edge_index, _ = subgraph(nodes_tensor, edge_index, relabel_nodes=True)

        graph_batch_nodes_set.update(nodes_tensor.tolist())
        graph_batch_latents, graph_batch_edge_index = inputs.to(device), subgraph_edge_index.to(device)
        graph_batch_x_next = sample(args, inputs.clone(), graph_batch_latents, model.gnn_denoiser_D, graph_batch_edge_index, dim=sample_dim) 
        
        assert graph_batch_x_next.shape[0] == len(nodes_tensor), f"subgraph {i} node num not match!"        
        x_next_full[nodes_tensor] = graph_batch_x_next

    logger.info(f'final set result is: {graph_batch_nodes_set}')
    logger.info(f'final set length is: {len(graph_batch_nodes_set)}')
    logger.info(f'x_next_full shape is: {x_next_full.shape}')

    x_next = x_next_full * 2 + mean.to(device)
    syn_data = x_next.float().cpu().numpy()
    
    if not os.path.exists(f"visualize/generated_latent_embeddings/{args.dataname}"):
        os.makedirs(f"visualize/generated_latent_embeddings/{args.dataname}")
        
    np.save(f"visualize/generated_latent_embeddings/{args.dataname}/syn_data.npy", syn_data)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    
    syn_df = pd.DataFrame(get_ori_data(syn_data, info, args.device, threshold=args.threshold))
    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()

    logger.info(f'Time: {end_time - start_time}')
    logger.info(f'Saving sampled data to {save_path}')


if __name__ == '__main__':

    args = get_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    
    main(args)
import os
import numpy as np
import torch
import argparse
import warnings
import time
from tqdm import tqdm
from dataset.latent_engine import get_input_train
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from ggsd.gnn_denoiser import Model, GGSD_Diffusion
from partition.SEPG.utils import PartitionTree
from partition.SEPN.prepare_nodeData import update_node
from partition.utils import extract_cluster_assignment, get_all_partitions 
from utils.config import get_args
from utils.tools import create_logger


def main(args): 
    device = args.device
    logger = create_logger(args.log_dir)
    logger.info(f'logging directory created at {args.log_dir}')

    latent, _, _, weight_path, info_with_edges = get_input_train(args)
    logger.info(f'latent device is: {latent.device}')

    edge_index = torch.tensor(info_with_edges['edge_index'], dtype=torch.long)
    logger.info(f'here edge_index shape is: {edge_index.shape}')
    logger.info(f'weight_path is: {weight_path}')

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    in_dim = latent.shape[1] 

    mean, std = latent.mean(0), latent.std(0)
    latent = (latent - mean) / 2
    train_data = latent

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

    train_graph_data = Data(x=train_data, edge_index=edge_index)
    partitions_k = partitions_per_layer[SE_k]
    subgraph_edges_list = []
    subgraph_node_mappings = [] 
    for subgraph_idx in partitions_k.keys():
        nodes_tensor = torch.tensor(partitions_k[subgraph_idx], dtype=torch.long)
        subgraph_edge_index, _ = subgraph(nodes_tensor, edge_index, relabel_nodes=True)
        subgraph_node_mappings.append(partitions_k[subgraph_idx])
        subgraph_edges_list.append(subgraph_edge_index)
        
    num_epochs = args.epochs

    gnn_denoiser = GGSD_Diffusion(in_dim, 1024).to(device)

    num_params = sum(p.numel() for p in gnn_denoiser.parameters())
    logger.info(f'the number of parameters is: {num_params}')

    model = Model(gnn_denoiser = gnn_denoiser, hid_dim = latent.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        
        pbar = tqdm(partitions_k.keys(), total=len(partitions_k.keys()))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for subgraph_idx in pbar:
            nodes_tensor = torch.tensor(partitions_k[subgraph_idx], dtype=torch.long)
            inputs = train_data[partitions_k[subgraph_idx]]
            subgraph_edge_index, _ = subgraph(nodes_tensor, edge_index, relabel_nodes=True)
            
            inputs = inputs.to(device)
            subgraph_edge_index = subgraph_edge_index.to(device)
            loss = model(inputs, subgraph_edge_index)
            loss = loss.mean()
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            if not os.path.exists(f'{weight_path}'):
                os.makedirs(f'{weight_path}')

            torch.save(model.state_dict(), f'{weight_path}/{args.method}.pt')

        else:
            patience += 1
            if patience == 500:
                logger.info('Early stopping')
                break

        if epoch % 1000 == 0:
            if not os.path.exists(f'{weight_path}'):
                os.makedirs(f'{weight_path}')
            torch.save(model.state_dict(), f'{weight_path}/{args.method}_{epoch}.pt')

    end_time = time.time()
    logger.info(f'Time: {end_time - start_time}')


if __name__ == '__main__':

    args = get_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    
    main(args)
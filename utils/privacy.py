import torch
import torch.nn.functional as F


def edge_index_to_adj_matrix(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device="cuda")
    edge_i, edge_j = edge_index[0], edge_index[1]
    adj_matrix[edge_i, edge_j] = 1.0
    
    return adj_matrix


def compute_info_nce_loss(subgraph_r, subgraph_t, temperature):
    device = subgraph_t.device
    batch_size = subgraph_r.size(0)
    subgraph_r_norm = F.normalize(subgraph_r, dim=1)
    subgraph_t_norm = F.normalize(subgraph_t, dim=1)
    
    logits = torch.matmul(subgraph_t_norm, subgraph_r_norm.T) / temperature
    labels = torch.arange(batch_size).to(device)
    loss = F.cross_entropy(logits, labels)
    return -loss
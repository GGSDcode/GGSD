import torch

def extract_cluster_assignment(T, tree_depth):
    layer_idx = [0]

    for layer in range(tree_depth+1):
        layer_nodes = [i for i, n in T.items() if n['depth']==layer]
        layer_idx.append(layer_idx[-1] + len(layer_nodes))
    
    interLayerEdges = [[] for i in range(tree_depth+1)]

    for i, n in T.items():
        if n['depth'] == 0:
            continue
        n_idx = n['ID'] - layer_idx[n['depth']]
        c_base = layer_idx[n['depth']-1]
        interLayerEdges[n['depth']].extend([(n_idx, c-c_base) for c in n['children']])
    
    return interLayerEdges[1:]


def convert_to_adjacency_matrix(S_edge_index, num_partitions):
    adj_matrix = torch.zeros((num_partitions, num_partitions), dtype=torch.float32)
    
    for i in range(S_edge_index.shape[1]):  
        target_partition = S_edge_index[0, i].item()  
        source_partition = S_edge_index[1, i].item()  
        adj_matrix[target_partition, source_partition] = 1  
    
    return adj_matrix


def extract_partitions(S_edge_index, num_partitions, old_partition_dict):
    new_partition_dict = {i: [] for i in range(num_partitions)}
    
    for i in range(S_edge_index.shape[1]):  
        target_partition = S_edge_index[0, i].item()
        node = S_edge_index[1, i].item()
        if old_partition_dict==None:
            new_partition_dict[target_partition].append(node)
        else:
            new_partition_dict[target_partition].extend(old_partition_dict[node])
    
    return new_partition_dict


def get_all_partitions(all_S_edge_indices):
    partitions_per_layer = []
    
    for i, S_edge_index in enumerate(all_S_edge_indices):
        num_partitions = len(set(S_edge_index[0].numpy()))
        if i==0:
            old_partition_dict = None
        else:
            old_partition_dict = partitions_per_layer[-1]
        new_partition_dict = extract_partitions(S_edge_index, num_partitions, old_partition_dict=old_partition_dict)
        partitions_per_layer.append(new_partition_dict)
    
    return partitions_per_layer
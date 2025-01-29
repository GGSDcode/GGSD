import torch
import torch.nn.functional as F
from torch_geometric.datasets import Amazon
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset_name):
    if 'Amazon' in dataset_name:
        dataset = Amazon('tmp/graphDataset/Amazon', name=dataset_name.split('_')[-1])
    else:
        dataset_path = f'tmp/graphDataset/{dataset_name}/processed/data.pt'
        dataset = torch.load(dataset_path)

    data = dataset[0]
    return data


def one_hot_encode(labels, num_classes):
    
    return F.one_hot(labels, num_classes=num_classes).float()


def get_dataloaders(data, batch_size=128, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    num_classes = data.y.max().item() + 1
    print(f"Number of classes: {num_classes}")

    y_one_hot = one_hot_encode(data.y, num_classes=num_classes)
    print(f"One-hot encoded y shape: {y_one_hot.shape}")

    combined_features = torch.cat((data.x, y_one_hot), dim=1)
    print(f"Combined features shape: {combined_features.shape}")

    dataset = TensorDataset(combined_features)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def weighted_bce_loss(recon_x, x, pos_weight):
    loss = - (pos_weight * x * torch.log(recon_x + 1e-10) + (1 - x) * torch.log(1 - recon_x + 1e-10))
    return loss.sum()


def loss_function(X_dim, recon_x, x, mu, logvar, beta):
    recon_x[:, :X_dim] = torch.sigmoid(recon_x[:, :X_dim])
    recon_loss_X = F.binary_cross_entropy(recon_x[:, :X_dim], x[:, :X_dim], reduction='sum')
    recon_loss_y = F.cross_entropy(recon_x[:, X_dim:], x[:, X_dim:].argmax(dim=1), reduction='sum')
    recon_loss = recon_loss_X + recon_loss_y
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta*kl_loss, recon_loss, kl_loss


def prepare_features(data, num_indices, cat_indices):
    x_num = data.x[:, num_indices]
    x_cat = data.x[:, cat_indices].long()

    if x_num.size(1) > 0:
        scaler = StandardScaler()
        x_num_np = x_num.detach().cpu().numpy()
        x_num_standardized = scaler.fit_transform(x_num_np)
        x_num_processed = torch.tensor(x_num_standardized, dtype=torch.float32).to(data.x.device)
        
        min_val = x_num_processed.min(dim=0)[0]
        max_val = x_num_processed.max(dim=0)[0]
        x_num_processed = (x_num_processed - min_val) / (max_val - min_val + 1e-8)
    else:
        x_num_processed = torch.empty((data.x.size(0), 0), device=data.x.device)

    x_cat_processed_list = []
    for i in range(x_cat.size(1)):
        col = x_cat[:, i]
        num_classes = col.max().item() + 1
        col_one_hot = F.one_hot(col, num_classes=num_classes).float()
        x_cat_processed_list.append(col_one_hot)
    
    x_cat_processed = torch.cat(x_cat_processed_list, dim=1) if x_cat_processed_list else torch.empty((data.x.size(0), 0), device=data.x.device)

    x_combined = torch.cat([x_num_processed, x_cat_processed], dim=1)

    num_classes_y = data.y.max().item() + 1
    y_one_hot = F.one_hot(data.y, num_classes=num_classes_y).float().to(data.x.device)

    combined_features = torch.cat([x_combined, y_one_hot], dim=1)
    print(f"combined_features shape: {combined_features.shape}")
    return combined_features
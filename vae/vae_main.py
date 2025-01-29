import argparse
import warnings
import os
from tqdm import tqdm
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from model import VAE
from utils import load_dataset, loss_function, get_dataloaders, one_hot_encode, prepare_features


def kl_warmup_scheduler(epoch, total_epochs, min_beta, max_beta):
    min_beta = min_beta
    max_beta = max_beta
    return min_beta + (epoch / total_epochs) * (max_beta - min_beta)


def main(args):
    min_beta = args.min_beta
    max_beta = args.max_beta
    lambd = args.lambd
    dataname = args.dataname
    data = load_dataset(dataname)

    device = args.device
    data = data.to(device)
    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=args.batch_size)
    
    ckpt_dir = f'{args.ckpt_dir}/{dataname}'
    os.makedirs(ckpt_dir, exist_ok=True)  

    model_save_path = os.path.join(ckpt_dir, 'model.pt')
    encoder_save_path = os.path.join(ckpt_dir, 'encoder.pt')
    decoder_save_path = os.path.join(ckpt_dir, 'decoder.pt')

    input_dim = data.x.shape[1] + one_hot_encode(data.y, data.y.max().item() + 1).shape[1]
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim

    print(f"Input Dim: {input_dim}, Latent Dim: {latent_dim}, Hidden Dim: {hidden_dim}")

    model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    epochs = args.epochs
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        beta = kl_warmup_scheduler(epoch, epochs+1, min_beta, max_beta)
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0

        for batch in train_loader:

            x = batch[0].to(device).float()  

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            
            loss, recon_loss, kl_loss = loss_function(data.x.shape[1], recon_x, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_recon = train_recon_loss / len(train_loader.dataset)
        avg_train_kl = train_kl_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device).float()
                recon_x, mu, logvar = model(x)
                loss, recon_loss, kl_loss = loss_function(data.x.shape[1], recon_x, x, mu, logvar, beta=1.0)

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_recon = val_recon_loss / len(val_loader.dataset)
        avg_val_kl = val_kl_loss / len(val_loader.dataset)

        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch}/{epochs} | '
              f'Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}) | '
              f'Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f})')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"the best model has been saved，val loss is: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"no improvement. patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            break


    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_loss = 0.0
    test_recon_loss = 0.0
    test_kl_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device).float()
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = loss_function(data.x.shape[1], recon_x, x, mu, logvar, beta=1.0)

            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_recon = test_recon_loss / len(test_loader.dataset)
    avg_test_kl = test_kl_loss / len(test_loader.dataset)

    print(f'test loss: {avg_test_loss:.4f} (recon loss: {avg_test_recon:.4f}, KL loss: {avg_test_kl:.4f})')

    encoder = model.encoder
    decoder = model.decoder

    torch.save(encoder.state_dict(), encoder_save_path)
    torch.save(decoder.state_dict(), decoder_save_path)
    print("encoder and decoder both saved.")


    combined_features = prepare_features(data)
    
    dataset = TensorDataset(combined_features)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    latent_features = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device).float()
            mu, logvar = encoder(x)

            latent = mu
            latent_features.append(latent.cpu())

    latent_features = torch.cat(latent_features, dim=0)
    print(f"Latent features shape: {latent_features.shape}")
    print(f"latent features: ", latent_features)
    np.save(f'{ckpt_dir}/train_z.npy', latent_features)

    print('Successfully save pretrained embeddings!')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='VAE training')

    parser.add_argument('--dataname', type=str, default='Cora', help='graph dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='batchsize of training')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='dim of latent space')
    parser.add_argument('--hidden_dim', type=int, default=512, help='dim of hidden layer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 Norm')
    parser.add_argument('--weight_dir', type=str, default='vae/weight', help='save weights')
    parser.add_argument('--patience', type=int, default=1000, help='')
    parser.add_argument('--max_beta', type=float, default=1.0, help='Maximum Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-1, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    print(f"using device: {args.device}")

    main(args)

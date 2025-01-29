import argparse


def get_args(default_args=None):

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--rho", type=int, default=7, help='rho')

    parser.add_argument('--steps', type=int, default=50, help='NFEs.')

    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    parser.add_argument('--epochs', type=int, default=1000, help='epochs')

    parser.add_argument('--patience', type=int, default=1000, help='patience')

    parser.add_argument("--SE_k", type=int, default=4, help='SE_k')
    
    parser.add_argument("--tree_depth", type=int, default=5, help='tree_depth')

    parser.add_argument("--threshold", type=float, default=0.5, help='threshold')

    parser.add_argument("--S_min", type=float, default=0, help='S_min')

    parser.add_argument("--S_max", type=float, default=float('inf'), help='S_max')

    parser.add_argument("--S_churn", type=float, default=1, help='S_churn')

    parser.add_argument("--S_noise", type=float, default=1, help='S_noise')

    parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')

    parser.add_argument("--SIGMA_MAX", type=float, default=80, help='SIGMA_MAX')

    parser.add_argument("--SIGMA_MIN", type=float, default=2e-3, help='SIGMA_MIN')

    parser.add_argument('--min_beta', type=float, default=0.1, help='Minimum beta')

    parser.add_argument('--max_beta', type=float, default=1.0, help='Maximum beta')

    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 Norm')

    parser.add_argument("--seed", type=int, default=42, help="seed for experiments")

    parser.add_argument("--partition_strategy", type=str, default='structural_entropy')

    parser.add_argument('--dataname', type=str, default='Cora', help='graph dataset name')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--latent_dim', type=int, default=128, help='dim of latent space')

    parser.add_argument('--hidden_dim', type=int, default=512, help='dim of hidden layer')

    parser.add_argument('--method', type=str, default='GGSD', help='Method: GGSD or VAE.')
    
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')

    parser.add_argument('--weight_dir', type=str, default='vae/weight', help='save weights')

    parser.add_argument("--log_dir", type=str, default="./log_dir", help='logging directory')

    parser.add_argument('--sample_step', type=int, default=2000, help='frequency of sampling')

    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')

    parser.add_argument("--graph_condition", action='store_true', default=True, help='Whether use graph condition')

    parser.add_argument("--CL_guidance", action='store_true', default=True, help='Whether use contrastive guidance')

    parser.add_argument('--var_type', type=str, default='fixedsmall', choices=['fixedlarge', 'fixedsmall'], help='variance type')

    parser.add_argument('--mean_type', type=str, default='epsilon', choices=['xprev', 'xstart', 'epsilon'], help='predict variable')

    args = parser.parse_args(default_args)

    return args
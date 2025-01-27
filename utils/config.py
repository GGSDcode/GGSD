import argparse

def get_args(default_args=None):

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--partition_strategy", type=str, default='structural_entropy')
    parser.add_argument("--graph_condition", action='store_true', default=True, help='Whether use graph condition')
    parser.add_argument("--CL_guidance", action='store_true', default=True, help='Whether use contrastive guidance')
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument("--seed", type=int, default=42, help="seed for experiments")
    parser.add_argument("--SIGMA_MIN", type=float, help='SIGMA_MIN')
    parser.add_argument("--SIGMA_MAX", type=float, help='SIGMA_MAX')
    parser.add_argument("--rho", type=float, help='rho')
    parser.add_argument("--S_churn", type=float, help='S_churn')
    parser.add_argument("--S_min", type=float, help='S_min')
    parser.add_argument("--S_max", type=float, help='S_max')
    parser.add_argument("--S_noise", type=float, help='S_noise')

    args = parser.parse_args(default_args)

    return args
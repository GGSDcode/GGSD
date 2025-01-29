import os
import torch
import random
import numpy as np
from utils.config import get_args
from execute import execute_function


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    if not args.save_path:
        args.save_path = f'synthetic/{args.dataname}/{args.method}.csv'
        if not os.path.exists(f'synthetic/{args.dataname}'):
            os.makedirs(f'synthetic/{args.dataname}')

    main_fn = execute_function(args.method, args.mode)

    main_fn(args)
import argparse
import importlib

def execute_function(method, mode):
    if method == 'vae':
        mode = 'vae_main'
    if method == 'GGSD':
        if mode == 'train':
            mode = 'train_GGSD'
        else:
            mode = 'inference_GGSD'

    if method == 'vae':
        module_name = f"vae.{mode}"
    elif method == 'GGSD':
        module_name = f"{mode}"

    try:
        train_module = importlib.import_module(module_name)
        train_function = getattr(train_module, 'main')
    except ModuleNotFoundError:
        print(f"Module {module_name} not found.")
        exit(1)
    except AttributeError:
        print(f"Function 'main' not found in module {module_name}.")
        exit(1)
    return train_function
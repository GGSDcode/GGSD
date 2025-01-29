import logging
import torch.distributed as dist

def create_logger(logging_dir=None, logging_file=None, ddp=False):

    if not ddp or (ddp and dist.get_rank() == 0):
        if logging_file is not None:
            file_handler = [logging.FileHandler(logging_file)]
        elif logging_dir is not None:
            file_handler = [logging.FileHandler(f"{logging_dir}/log.txt")]
        else:
            file_handler = []
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler()] + file_handler,
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

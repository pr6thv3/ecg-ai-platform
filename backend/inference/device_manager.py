import logging
import torch

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Detects the best available device (CUDA or CPU).
    Logs the selection on startup.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Device Manager: Selected CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("Device Manager: CUDA not available. Selected CPU.")
    return device

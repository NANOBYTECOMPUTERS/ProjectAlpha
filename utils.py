import time
import cv2
import torch
from logger import log_error, setup_logging
from config import cfg



def cleanup_resources(visualization=None):
    try:
        if visualization and visualization.running:
            visualization.queue.put(None)
            time.sleep(0.1)
            visualization.cleanup()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()
    except Exception as e:
        log_error(f"Error during cleanup: {e}")

_torch_device_cache = None
def get_torch_device():
    global _torch_device_cache
    if _torch_device_cache is None:
        ai_device = str(cfg.ai_device).strip().lower()
        if cfg.ai_enable_amd:
            _torch_device_cache = torch.device(f'hip:{ai_device}')
        elif 'cpu' in ai_device:
            _torch_device_cache = torch.device('cpu')
        elif ai_device.isdigit():
            _torch_device_cache = torch.device(f'cuda:{ai_device}')
        else:
            _torch_device_cache = torch.device('cuda:0')
    return _torch_device_cache
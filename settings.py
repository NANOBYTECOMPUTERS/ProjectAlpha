from config import cfg
from utils import log_error

CLASS_NAMES = {0: 'player'}

def validate_config():
    numeric_configs = [
        ('mouse_speed', cfg.mouse_speed, lambda x: x > 0),
        ('detection_window_width', cfg.detection_window_width, lambda x: x > 0),
        ('detection_window_height', cfg.detection_window_height, lambda x: x > 0),
    ]
    for name, value, validator in numeric_configs:
        if not validator(value):
            raise ValueError(f"Invalid {name}: {value}")

def warnings():
    if ".pt" in cfg.ai_model_name:
        print("FYI: Export `.engine` for better performance!")
    if cfg.show_window:
        print("Debug using resources.")
    if cfg.bettercam_capture_fps >= 120:
        print("WARNING: High FPS can affect behavior (shaking).")
    if cfg.detection_window_width >= 600 or cfg.detection_window_height >= 600:
        print("WARNING: Large capture window might hurt performance.")
    if cfg.ai_conf <= 0.15:
        print("WARNING: Low `ai_conf` may lead to false positives.")
    if cfg.ai_disable_tracker:
        print("Disabling tracker will break performance.")
    if not (cfg.mouse_ghub or cfg.arduino_move or cfg.arduino_shoot or cfg.mouse_rzr):
        print("Win32 input set.")
    if cfg.mouse_ghub and not (cfg.arduino_move or cfg.arduino_shoot):
        print("WARNING: gHub might cause issues.")
    selected_methods = sum([cfg.arduino_move, cfg.mouse_ghub, cfg.mouse_rzr])
    if selected_methods > 1:
        raise ValueError("Only one input method should be selected.")

def run_checks():
    import torch
    import os
    if not torch.cuda.is_available():
        raise RuntimeError("Install PyTorch with CUDA support.")
    capture_methods = sum([cfg.mss_capture, cfg.bettercam_capture, cfg.obs_capture])
    if capture_methods != 1:
        raise RuntimeError("Exactly one capture method must be enabled.")
    model_path = os.path.join("models", cfg.ai_model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{cfg.ai_model_name}' not found.")
    warnings()
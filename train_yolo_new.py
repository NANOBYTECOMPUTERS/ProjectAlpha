from ultralytics import YOLO
import os
from config import cfg  # Import your config module
from logger import log_error  # Import your logger for error reporting

# Configuration from cfg
INITIAL_MODEL = "yolo12n.pt"  # Pretrained nano model
DATA_YAML = cfg.data_yaml  # Path to data.yaml from config
DEVICE = cfg.device  # GPU device from config
IMGSZ = cfg.train_size  # Training image size from config
BASE_DIR = cfg.train_base_dir  # Base directory for training runs from config

#training stages
TRAINING_STAGES = [
    # Stage 1: Initial broad training
    {
        "epochs": cfg.train_epochs,  # Number of epochs from config
        "batch": cfg.train_batch,  # Batch size from config
        "lr0": cfg.train_lr0,  # Initial learning rate from config
        "lrf": cfg.train_lrf,  # Final learning rate factor from config
        "freeze": cfg.train_freeze,  # Layers to freeze from config
        "box": cfg.train_box,  # Box loss weight from config
        "iou": cfg.train_iou,  # IoU threshold from config
        "patience": cfg.train_patience,  # Early stopping patience from config
        "weight_decay": cfg.train_weight_decay,  # Weight decay from config
        "dropout": cfg.train_dropout,  # Dropout rate from config
    },
    # Stage 2: Refine with lower LR
    {
        "epochs": cfg.train_epochs // 2,  # Half epochs for refinement
        "batch": cfg.train_batch,
        "lr0": cfg.train_lr0 / 5,  # Reduced LR
        "lrf": cfg.train_lrf,
        "freeze": cfg.train_freeze // 2,  # Less freezing
        "box": cfg.train_box + 0.5,  # Increase box focus
        "iou": cfg.train_iou + 0.05,  # Stricter IoU
        "patience": cfg.train_patience,
        "weight_decay": cfg.train_weight_decay *1.25,  # Slightly more regularization
        "dropout": cfg.train_dropout,
    },
    # Stage 3: Polish with regularization
    {
        "epochs": cfg.train_epochs // 3,  # Shorter polish stage
        "batch": cfg.train_batch,
        "lr0": cfg.train_lr0 / 7.5,  # Very low LR
        "lrf": cfg.train_lrf / 2,
        "freeze": 0,  # Full model
        "box": cfg.train_box + 1.0,  # Stronger box focus
        "iou": cfg.train_iou + 0.08,  # strict IoU
        "patience": cfg.train_patience // 2,
        "weight_decay": cfg.train_weight_decay * 2,  # Increased regularization
        "dropout": cfg.train_dropout + 0.1,  # Add dropout
    }
]

def train_stage(model_path, stage_idx, params):
    """Train a single stage and return the best model path."""
    run_name = f"train_stage_{stage_idx}"
    model = YOLO(model_path)
    
    # Ensure stage directory exists
    stage_dir = os.path.join(BASE_DIR, run_name)
    os.makedirs(stage_dir, exist_ok=True)
    
    try:
        # Train with specified parameters
        results = model.train(
            data=DATA_YAML,
            epochs=params["epochs"],
            imgsz=IMGSZ,
            batch=params["batch"],
            lr0=params["lr0"],
            lrf=params["lrf"],
            freeze=params["freeze"],
            box=params["box"],
            iou=params["iou"],
            patience=params["patience"],
            weight_decay=params["weight_decay"],
            dropout=params["dropout"],
            device=DEVICE,
            project=BASE_DIR,
            name=run_name,
            exist_ok=True,
            resume=False,  # Fresh start each stage
            save=True,
            verbose=True
        )
    except Exception as e:
        log_error(f"Error during training stage {stage_idx}: {e}")
        raise
    
    # Best model path after training
    best_model_path = os.path.join(BASE_DIR, run_name, "weights", "best.pt")
    if not os.path.exists(best_model_path):
        log_error(f"Best model not found at {best_model_path} after stage {stage_idx}")
        raise FileNotFoundError(f"Best model not saved at {best_model_path}")
    
    return best_model_path

def main():
    """Run a series of training stages, transferring weights."""
    current_model = INITIAL_MODEL
    
    # Ensure base directory exists
    os.makedirs(BASE_DIR, exist_ok=True)
    
    for idx, params in enumerate(TRAINING_STAGES, start=1):
        print(f"\nStarting Stage {idx} with {current_model}")
        try:
            current_model = train_stage(current_model, idx, params)
            print(f"Stage {idx} completed. Best model saved at: {current_model}")
        except Exception as e:
            log_error(f"Stage {idx} failed: {e}")
            return
    
    # Final model export with error handling
    final_model = YOLO(current_model)
    export_path = os.path.join(BASE_DIR, "your_new_model.pt")
    
    try:
        # Ensure export directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Export to ONNX
        result = final_model.export(
            format="engine",
            imgsz=IMGSZ,
            device=DEVICE
        )
        if not os.path.exists(result):
            log_error(f"Export failed: No file found at {result}")
            raise FileNotFoundError(f"Export did not produce {result}")
        print(f"\nFinal trained model exported to: {result}")
    except Exception as e:
        log_error(f"Error exporting final model: {e}")
        print(f"\nExport failed: {e}")
    else:
        log_error(f"Model export successful: {result}")

if __name__ == "__main__":
    main()
from ultralytics import YOLO
import os
from config import cfg
# Configuration
PRETRAINED_MODEL = cfg.finetune_pretrained_model  # Pretrained model path from config
DATA_YAML = cfg.finetune_data_yaml  # Path to data.yaml from config
DEVICE = cfg.device  # GPU device from config
IMGSZ = cfg.finetune_size  # Fine-tuning image size from config
BASE_DIR = cfg.finetune_base_dir  # Base directory for fine-tuning runs from config
# Define fine-tuning stages
FINETUNE_STAGES = [
    # Stage 1: Initial refinement with moderate LR
    {
        "epochs": cfg.finetune_epochs,  # Number of epochs from config
        "batch": cfg.finetune_batch,  # Batch size from config
        "lr0": cfg.finetune_lr0,  # Initial learning rate from config
        "lrf": cfg.finetune_lrf,  # Final learning rate factor from config
        "freeze": cfg.finetune_freeze,  # Layers to freeze from config
        "box": cfg.finetune_box,  # Box loss weight from config
        "iou": cfg.finetune_iou,  # IoU threshold from config
        "patience": cfg.finetune_patience,  # Early stopping patience from config
        "weight_decay": cfg.finetune_weight_decay,  # Weight decay from config
        "dropout": cfg.finetune_dropout,  # Dropout rate from config
    },
    # Stage 2: Stronger box focus, lower LR
    {
        "epochs": cfg.finetune_epochs // 2,  # Half epochs
        "batch": cfg.finetune_batch,
        "lr0": cfg.finetune_lr0 / 5,  # Reduced LR
        "lrf": cfg.finetune_lrf,
        "freeze": cfg.finetune_freeze // 2,  # Less freezing
        "box": cfg.finetune_box + 0.5,  # Increase box precision
        "iou": cfg.finetune_iou + 0.025,  # Stricter IoU
        "patience": cfg.finetune_patience,
        "weight_decay": cfg.finetune_weight_decay,
        "dropout": cfg.finetune_dropout,
    },
    # Stage 3: Polish with regularization
    {
        "epochs": cfg.finetune_epochs // 3,  # Shorter polish stage
        "batch": cfg.finetune_batch,
        "lr0": cfg.finetune_lr0 / 7.5,  # Very low LR
        "lrf": cfg.finetune_lrf / 1.5,
        "freeze": 0,  # Full model
        "box": cfg.finetune_box + 1.0,  # Max box focus
        "iou": cfg.finetune_iou + 0.05,  # Very strict IoU
        "patience": cfg.finetune_patience // 2,
        "weight_decay": cfg.finetune_weight_decay * 1.5,  # Stronger regularization
        "dropout": cfg.finetune_dropout + 0.05,  # Add dropout
    }
]

def finetune_stage(model_path, stage_idx, params):
    run_name = f"finetune_stage_{stage_idx}"
    model = YOLO(model_path)
    # Fine-tune with specified parameters
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
    # Best model path after fine-tuning
    best_model_path = os.path.join(BASE_DIR, run_name, "weights", "best.pt")
    return best_model_path

def main():
    """Run a series of fine-tuning stages, transferring weights."""
    current_model = PRETRAINED_MODEL
    
    for idx, params in enumerate(FINETUNE_STAGES, start=1):
        print(f"\nStarting Stage {idx} with {current_model}")
        current_model = finetune_stage(current_model, idx, params)
        print(f"Stage {idx} completed. Best model saved at: {current_model}")
        
    # Final model export
    final_model = YOLO(current_model)
    export_path = os.path.join(BASE_DIR, "finetuned-best.pt")
    final_model.export(format="engine", imgsz=IMGSZ)
    print(f"\nFinal fine-tuned model exported to: {export_path}")

if __name__ == "__main__":
    main()
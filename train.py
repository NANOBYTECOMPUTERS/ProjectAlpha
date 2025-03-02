import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from config import cfg
from datetime import datetime
from tqdm import tqdm
import cupy as cp

class MouseMLP(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.fc1 = nn.Linear(10, 512)  # Increased width for more capacity
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)  # Slightly higher dropout for regularization
        self.device = device
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class TrainMLP:
    def __init__(self):
        self.m_spd = cfg.mouse_speed
        self.tbot_box_size = cfg.tbot_box_size
        self.m_spd_bst = cfg.mouse_speed_boost
        self.m_bst_thresh = cfg.mouse_boost_threshold
        self.max_target_speed = cfg.max_target_speed
        self.body_y_offset = cfg.body_y_offset
        self.scr_w = cfg.detection_window_width
        self.scr_h = cfg.detection_window_height
        self.center_x = self.scr_w / 2.0
        self.center_y = self.scr_h / 2.0
        self.scale_x = self.m_spd / self.scr_w
        self.scale_y = self.m_spd / self.scr_h
        self.frame_time = 1 / cfg.bettercam_capture_fps
        self.use_mlp = cfg.mouse_mlp
        self.mlp_samples = cfg.mlp_samples
        self.mlp_epochs = cfg.mlp_epochs
        self.checkpoint_freq = cfg.checkpoint_freq
        self.patience = cfg.patience
        self.initial_lr = cfg.initial_lr
        self.warmup_epochs = cfg.warmup_epochs
        self.min_lr = cfg.min_lr
        self.lr_factor = cfg.lr_factor
        self.lr_patience = cfg.lr_patience
        self.batch_size = cfg.batch_size
        self.load_existing_mlp = cfg.load_existing_mlp
        self.smoothing_factor = cfg.smoothing_factor
        
        # Enhanced device selection for quality
        self.device = self.get_torch_device()
        
        self.run_base_dir = "mlp_runs"
        os.makedirs(self.run_base_dir, exist_ok=True)
        run_number = max([int(d[3:]) for d in os.listdir(self.run_base_dir) if d.startswith("run") and d[3:].isdigit()] + [0]) + 1
        self.run_dir = os.path.join(self.run_base_dir, f"run{run_number}")
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"Starting new training run in {self.run_dir}")

        self.mouse_mlp = MouseMLP(device=self.device)
        if self.load_existing_mlp and os.path.exists("mouse_mlp.pth"):
            self.mouse_mlp.load_state_dict(torch.load("mouse_mlp.pth"))
            print("Loaded existing model from 'mouse_mlp.pth'")

    def get_torch_device(self):
        # Align with alpha.py for compatibility and flexibility
        ai_device = str(cfg.ai_device).strip().lower()
        if cfg.ai_enable_amd and torch.cuda.is_available():
            return torch.device(f'hip:{ai_device}')
        elif 'cpu' in ai_device or not torch.cuda.is_available():
            return torch.device('cpu')
        elif ai_device.isdigit():
            return torch.device(f'cuda:{ai_device}')
        else:
            return torch.device('cuda:0')

    def generate_random_bounding_boxes(self, num_boxes=1):
        # Keep CuPy for fast, high-quality data generation
        w = cp.random.uniform(50, 200, size=num_boxes)
        h = cp.random.uniform(50, 200, size=num_boxes)
        x1 = cp.random.uniform(0, self.scr_w - w, size=num_boxes)
        y1 = cp.random.uniform(0, self.scr_h - h, size=num_boxes)
        x2 = x1 + w
        y2 = y1 + h
        boxes = cp.stack([x1, y1, x2, y2], axis=1)
        return boxes.get().astype(np.float32)

    def simulate_target_movement(self, target_x, target_y, prev_x, prev_y):
        vel_x = np.random.uniform(-self.max_target_speed, self.max_target_speed)  # Bidirectional velocity
        vel_y = np.random.uniform(-self.max_target_speed, self.max_target_speed)
        new_x = target_x + vel_x * self.frame_time
        new_y = target_y + vel_y * self.frame_time
        target_w = min(200, self.scr_w / 4)
        target_h = min(200, self.scr_h / 4)
        new_x = np.clip(new_x, 0, self.scr_w - target_w)
        new_y = np.clip(new_y, 0, self.scr_h - target_h)
        return new_x, new_y

    def calc_movement(self, target_x, target_y, prev_x, prev_y, smoothed_x=None, smoothed_y=None):
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        distance = np.hypot(offset_x, offset_y)
        speed_boost = self.m_spd_bst if distance < self.m_bst_thresh else 1.0
        move_x = offset_x * self.scale_x * speed_boost
        move_y = offset_y * self.scale_y * speed_boost

        if prev_x is not None and prev_y is not None:
            vel_x = (target_x - prev_x) / self.frame_time
            vel_y = (target_y - prev_y) / self.frame_time
            pred_x = target_x + vel_x * self.frame_time
            pred_y = target_y + vel_y * self.frame_time
            pred_offset_x = pred_x - self.center_x
            pred_offset_y = pred_y - self.center_y
            move_x = pred_offset_x * self.scale_x * speed_boost
            move_y = pred_offset_y * self.scale_y * speed_boost

        if smoothed_x is not None and smoothed_y is not None:
            move_x = self.smoothing_factor * move_x + (1 - self.smoothing_factor) * (smoothed_x - self.center_x)
            move_y = self.smoothing_factor * move_y + (1 - self.smoothing_factor) * (smoothed_y - self.center_y)

        return move_x, move_y

    def train(self):
        if not self.use_mlp:
            print("MouseMLP training skipped: 'mouse_mlp' is disabled in config.")
            return

        cfg.read(verbose=True)
        num_samples = self.mlp_samples
        print(f"Generating {num_samples} training samples...")
        boxes = self.generate_random_bounding_boxes(num_samples)
        
        inputs = []
        targets = []
        smoothed_x, smoothed_y = None, None

        for i in tqdm(range(len(boxes)), desc="Processing training data", unit="sample"):
            x1, y1, x2, y2 = boxes[i]
            target_w = x2 - x1
            target_h = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            target_x = center_x
            target_y = center_y + target_h * (self.body_y_offset - 0.5)

            if i == 0:
                prev_x, prev_y = target_x, target_y
            else:
                prev_x, prev_y = inputs[-1][0], inputs[-1][1]
                target_x, target_y = self.simulate_target_movement(target_x, target_y, prev_x, prev_y)

            move_x, move_y = self.calc_movement(target_x, target_y, prev_x, prev_y, smoothed_x, smoothed_y)
            smoothed_x = self.center_x + move_x if smoothed_x is None else self.smoothing_factor * (self.center_x + move_x) + (1 - self.smoothing_factor) * smoothed_x
            smoothed_y = self.center_y + move_y if smoothed_y is None else self.smoothing_factor * (self.center_y + move_y) + (1 - self.smoothing_factor) * smoothed_y

            inputs.append([target_x, target_y, target_w, target_h, prev_x, prev_y, 
                           self.m_spd, self.tbot_box_size, self.m_spd_bst, self.m_bst_thresh])
            targets.append([move_x, move_y])

        # Normalize inputs for better training stability
        inputs_np = np.array(inputs, dtype=np.float32)
        input_means = inputs_np.mean(axis=0)
        input_stds = inputs_np.std(axis=0) + 1e-6  # Avoid division by zero
        inputs_normalized = (inputs_np - input_means) / input_stds
        inputs = torch.tensor(inputs_normalized, dtype=torch.float32).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

        train_size = int(0.8 * num_samples)
        train_inputs, val_inputs = inputs[:train_size], inputs[train_size:]
        train_targets, val_targets = targets[:train_size], targets[train_size:]

        optimizer = torch.optim.AdamW(self.mouse_mlp.parameters(), lr=self.initial_lr, weight_decay=1e-4)  # Added weight decay
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience, min_lr=self.min_lr, verbose=True
        )

        best_val_loss = float('inf')
        epochs_no_improve = 0
        num_epochs = self.mlp_epochs

        train_losses = []
        val_losses = []
        train_maes = []
        val_maes = []
        learning_rates = []

        num_batches = (train_size + self.batch_size - 1) // self.batch_size
        for epoch in range(num_epochs):
            if epoch < self.warmup_epochs:
                lr = self.min_lr + (self.initial_lr - self.min_lr) * epoch / self.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            current_lr = optimizer.param_groups[0]['lr']

            self.mouse_mlp.train()
            epoch_train_loss = 0.0
            epoch_train_mae = 0.0

            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, train_size)
                batch_inputs = train_inputs[start_idx:end_idx]
                batch_targets = train_targets[start_idx:end_idx]

                optimizer.zero_grad()
                outputs = self.mouse_mlp(batch_inputs)
                loss = criterion(outputs, batch_targets)
                mae = mae_criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mouse_mlp.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += loss.item() * (end_idx - start_idx) / train_size
                epoch_train_mae += mae.item() * (end_idx - start_idx) / train_size

                del outputs, loss, mae
                torch.cuda.empty_cache()

            self.mouse_mlp.eval()
            with torch.no_grad():
                val_outputs = self.mouse_mlp(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_mae = mae_criterion(val_outputs, val_targets)
                max_error = torch.max(torch.abs(val_outputs - val_targets)).item()

            scheduler.step(val_loss)

            train_losses.append(epoch_train_loss)
            val_losses.append(val_loss.item())
            train_maes.append(epoch_train_mae)
            val_maes.append(val_mae.item())
            learning_rates.append(current_lr)

            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss.item():.4f}, "
                  f"Train MAE: {epoch_train_mae:.4f}, Val MAE: {val_mae.item():.4f}, "
                  f"Max Val Error: {max_error:.4f}, LR: {current_lr:.6f}")

            if (epoch + 1) % self.checkpoint_freq == 0:
                checkpoint_path = os.path.join(self.run_dir, f"mouse_mlp_epoch_{epoch + 1}.pth")
                torch.save({
                    'state_dict': self.mouse_mlp.state_dict(),
                    'input_means': input_means,
                    'input_stds': input_stds
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'state_dict': self.mouse_mlp.state_dict(),
                    'input_means': input_means,
                    'input_stds': input_stds
                }, "mouse_mlp.pth")
                print(f"Best model saved to mouse_mlp.pth with Val Loss: {best_val_loss:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            del val_outputs
            torch.cuda.empty_cache()

        self.plot_training_progress(train_losses, val_losses, train_maes, val_maes, learning_rates)
        print(f"Training completed. Best model saved to 'mouse_mlp.pth'. Run data saved in {self.run_dir}")

    def plot_training_progress(self, train_losses, val_losses, train_maes, val_maes, learning_rates):
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(train_maes, label='Train MAE')
        plt.plot(val_maes, label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Training and Validation Mean Absolute Error')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(learning_rates, label='Learning Rate', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Over Time')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "training_progress.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Training progress plot saved to {plot_path}")

if __name__ == "__main__":
    trainer = TrainMLP()
    trainer.train()
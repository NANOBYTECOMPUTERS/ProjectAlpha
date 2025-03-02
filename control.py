#control.py ---

import os
import time
import win32api
import win32con
import numpy as np
import torch
import torch.nn as nn
import cupy as cp
from config import cfg  # Updated to new config module
from input import Buttons  # Updated to new input module
from utils import log_error


class MouseMLP(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class Controller:
    def __init__(self):
        self.m_spd = cfg.mouse_speed
        self.tbot_box_size = cfg.tbot_box_size
        self.m_spd_bst = cfg.mouse_speed_boost
        self.m_bst_thresh = cfg.mouse_boost_threshold
        self.smoothing_factor = cfg.smoothing_factor
        self.use_neural_net = cfg.mouse_mlp
        self.frame_time = 1 / cfg.bettercam_capture_fps

        self.scr_w = cfg.detection_window_width
        self.scr_h = cfg.detection_window_height
        self.center_x = self.scr_w / 2.0
        self.center_y = self.scr_h / 2.0
        self.center_gpu = cp.array([self.center_x, self.center_y], dtype=cp.float32)
        self.scale_gpu = cp.array([self.m_spd / self.scr_w, self.m_spd / self.scr_h], dtype=cp.float32)

        self.button_pressed = False
        self.prev_x = None
        self.prev_y = None
        self.prev_time = None
        self.tbot_box = False

        self.hotkey_codes = [Buttons.KEY_CODES.get(key.strip()) for key in cfg.hotkey_targeting if Buttons.KEY_CODES.get(key.strip()) is not None]
        self.triggerbot_hotkey_code = Buttons.KEY_CODES.get(cfg.hotkey_triggerbot, win32con.VK_RBUTTON)

        if self.use_neural_net:
            self.device = 'cuda' if torch.cuda.is_available() and cfg.ai_device != 'cpu' else 'cpu'
            self.mouse_mlp = MouseMLP(device=self.device)  # Updated architecture
            if os.path.exists("mouse_mlp.pth"):
                checkpoint = torch.load("mouse_mlp.pth")
                self.mouse_mlp.load_state_dict(checkpoint['state_dict'])
                self.input_means = torch.tensor(checkpoint['input_means'], device=self.device)
                self.input_stds = torch.tensor(checkpoint['input_stds'], device=self.device)
                log_error("Loaded trained MouseMLP from 'mouse_mlp.pth' with normalization")
            self.mouse_mlp.eval()
            self.input_buffer = torch.zeros(10, dtype=torch.float32, device=self.device)

        self.move_buffer = cp.zeros(2, dtype=cp.float32)
        self._init_input_device()

    def _init_input_device(self):
        if cfg.mouse_ghub:
            from controls.ghub import GhubMouse
            self.input_device = GhubMouse()
            self.move_func = self.input_device.mouse_xy
            self.press_func = self.input_device.mouse_down
            self.release_func = self.input_device.mouse_up
        elif cfg.mouse_rzr:
            from controls.rzctl import RZControl, MOUSE_CLICK
            dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls", "rzctl.dll")
            if not os.path.exists(dll_path):
                raise FileNotFoundError(f"DLL not found: {dll_path}")
            self.input_device = RZControl(dll_path)
            if not self.input_device.initialize():
                raise RuntimeError("RZControl init failed.")
            self.move_func = lambda x, y: self.input_device.move_mouse(int(x), int(y), True)
            self.press_func = lambda: self.input_device.click_mouse(MOUSE_CLICK.LEFT_DOWN)
            self.release_func = lambda: self.input_device.click_mouse(MOUSE_CLICK.LEFT_UP)
        elif cfg.arduino_move or cfg.arduino_shoot:
            from controls.arduino import ArduinoMouse
            self.input_device = ArduinoMouse()
            self.move_func = self.input_device.move
            self.press_func = self.input_device.press
            self.release_func = self.input_device.release
        else:
            self.input_device = None
            self.move_func = lambda x, y: win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)
            self.press_func = lambda: win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self.release_func = lambda: win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def process_target(self, target_x, target_y, target_w, target_h, shooting_state):
        pred_x, pred_y = self.predict_next_position(target_x, target_y, time.time())
        if self.use_neural_net:
            move_x, move_y = self.calc_movement_neural(pred_x, pred_y, target_w, target_h)
        else:
            move_x, move_y = self.calc_movement(pred_x, pred_y)

        self.tbot_box = self.check_target_in_scope(target_x, target_y, target_w, target_h) if cfg.triggerbot else False
        self.tbot_box = cfg.force_click or self.tbot_box

        self.move_and_shoot(move_x, move_y, shooting_state)

    def predict_next_position(self, target_x, target_y, current_time):
        if self.prev_time is None:
            self.prev_x, self.prev_y = target_x, target_y
            self.prev_time = current_time
            return target_x, target_y

        dt = current_time - self.prev_time
        if dt <= 0:
            self.prev_x, self.prev_y = target_x, target_y
            self.prev_time = current_time
            return target_x, target_y

        vel_x = (target_x - self.prev_x) / dt
        vel_y = (target_y - self.prev_y) / dt
        pred_x = target_x + vel_x * self.frame_time
        pred_y = target_y + vel_y * self.frame_time

        self.prev_x, self.prev_y = target_x, target_y
        self.prev_time = current_time
        return pred_x, pred_y

    def calc_movement(self, target_x, target_y):
        target = cp.array([target_x, target_y], dtype=cp.float32)
        offset = target - self.center_gpu
        distance = cp.hypot(offset[0], offset[1])
        speed_boost = cp.where(distance < self.m_bst_thresh, self.m_spd_bst, 1.0)
        self.move_buffer[:] = offset * self.scale_gpu * speed_boost
        return float(self.move_buffer[0].get()), float(self.move_buffer[1].get())

    def calc_movement_neural(self, target_x, target_y, target_w, target_h):
        target_y_adj = target_y + cfg.body_y_offset * target_h
        inputs = [
            target_x, target_y_adj, target_w, target_h,
            self.prev_x if self.prev_x is not None else target_x,
            self.prev_y if self.prev_y is not None else target_y_adj,
            self.m_spd, self.tbot_box_size, self.m_spd_bst, self.m_bst_thresh
        ]
        self.input_buffer[:] = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            normalized_inputs = (self.input_buffer - self.input_means) / self.input_stds
            move = self.mouse_mlp(normalized_inputs)
        return float(move[0]), float(move[1])

    def move_and_shoot(self, move_x, move_y, shooting_state):
        triggerbot_hotkey_pressed = not cfg.triggerbot_hotkey or (win32api.GetAsyncKeyState(self.triggerbot_hotkey_code) & 0x8000)
        should_move = cfg.mouse_auto_aim or (shooting_state and triggerbot_hotkey_pressed)
        should_shoot = (cfg.triggerbot and self.tbot_box and triggerbot_hotkey_pressed) or (cfg.mouse_auto_aim and self.tbot_box)

        if should_move:
            try:
                self.move_func(move_x, move_y)
            except Exception as e:
                log_error(f"Error in move_mouse: {e}")

        if should_shoot and not self.button_pressed:
            try:
                self.press_func()
                self.button_pressed = True
                time.sleep(0.01)
            except Exception as e:
                log_error(f"Error in press_button: {e}")
        elif (not self.tbot_box or not shooting_state) and self.button_pressed:
            try:
                self.release_func()
                self.button_pressed = False
                time.sleep(0.01)
            except Exception as e:
                log_error(f"Error in release_button: {e}")

    def check_target_in_scope(self, target_x, target_y, target_w, target_h):
        half_w = target_w * self.tbot_box_size / 2.0
        half_h = target_h * self.tbot_box_size / 2.0
        x1, x2 = target_x - half_w, target_x + half_w
        y1, y2 = target_y - half_h, target_y + half_h
        return (self.center_x > x1 and self.center_x < x2 and 
                self.center_y > y1 and self.center_y < y2)

    def get_shooting_state(self):
        try:
            for code in self.hotkey_codes:
                if win32api.GetAsyncKeyState(code) & 0x8000:
                    return True
            return False
        except Exception as e:
            log_error(f"Error checking key state: {e}")
            return False
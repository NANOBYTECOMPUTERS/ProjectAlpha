import queue
import threading
import time
import cv2
import win32gui
import win32api
import win32con
import os
import numpy as np
import tkinter as tk
from config import cfg
from input import Buttons
from utils import log_error
import supervision as sv

SCREENSHOT_DIRECTORY = "screenshots"
CIRCLE_RADIUS = 5

class Visualization(threading.Thread):
    def __init__(self):
        os.makedirs(SCREENSHOT_DIRECTORY, exist_ok=True)
        self.show_visuals = cfg.show_window or cfg.show_overlay
        if not self.show_visuals:
            return

        super().__init__()
        self.queue = queue.Queue(maxsize=1)
        self.daemon = True
        self.name = 'Visualization'
        self.image = None
        self.screenshot_taken = False
        self.screenshot_lock = threading.Lock()
        self.interpolation = cv2.INTER_NEAREST if cfg.show_window else None
        self.running = True
        self.capture = None  # Reference to Capture instance
        self.bounding_box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
        self.label_annotator = sv.LabelAnnotator(color=sv.Color.RED, text_color=sv.Color.WHITE, 
                                                 text_scale=0.5, text_thickness=1, text_padding=5, 
                                                 text_position=sv.Position.TOP_LEFT)

        if cfg.show_overlay:
            self.root = tk.Tk()
            self.root.title("Overlay")
            self.root.attributes('-transparent', True)
            self.root.geometry(f"{cfg.detection_window_width}x{cfg.detection_window_height}")
            self.canvas = tk.Canvas(self.root, width=cfg.detection_window_width, height=cfg.detection_window_height)
            self.canvas.pack()

        self.start()

    def run(self):
        if cfg.show_window:
            self.spawn_debug_window()
        while self.running:
            try:
                item = self.queue.get_nowait()
                if item is None:
                    continue
                image, detections = item if isinstance(item, tuple) else (item, sv.Detections.empty())
                if image is not None:
                    self.image = image
                    self.handle_screenshot()
                    if cfg.show_window:
                        self.display_debug_window(detections)
                    if cfg.show_overlay:
                        self.display_overlay(detections)
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                log_error("Error in visualization loop", e)

    def set_capture(self, capture):
        self.capture = capture

    def display_debug_window(self, detections):
        if self.image is None:
            log_error("No image available for display")
            return

        display_img = self.image.copy()
        log_error(f"Image shape: {display_img.shape}, Scale percent: {cfg.debug_window_scale_percent}")

        if cfg.debug_window_scale_percent != 100:
            scale_percent = cfg.debug_window_scale_percent
            height = int(display_img.shape[0] * scale_percent / 100)
            width = int(display_img.shape[1] * scale_percent / 100)
            display_img = cv2.resize(display_img, (width, height), interpolation=self.interpolation)

        if cfg.show_boxes:
            display_img = self.annotate_with_supervision(display_img, detections)


        cv2.imshow(cfg.debug_window_name, display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()


    def display_overlay(self, detections):
        self.canvas.delete("all")
        if cfg.overlay_show_boxes and len(detections.xyxy) > 0:
            for box in detections.xyxy:
                x1, y1, x2, y2 = box
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
        self.root.update()

    def handle_screenshot(self):
        screenshot_key = Buttons.KEY_CODES.get(cfg.debug_window_screenshot_key)
        if screenshot_key and win32api.GetAsyncKeyState(screenshot_key) & 0x8000 and not self.screenshot_taken:
            with self.screenshot_lock:
                self.screenshot_taken = True
                if self.image is not None:
                    threading.Thread(target=self.save_screenshot, args=(self.image.copy(),), daemon=True).start()
        elif screenshot_key and not (win32api.GetAsyncKeyState(screenshot_key) & 0x8000):
            self.screenshot_taken = False

    def save_screenshot(self, image):
        filename = os.path.join(SCREENSHOT_DIRECTORY, f"{time.time()}.jpg")
        try:
            cv2.imwrite(filename, image)
        except Exception as e:
            log_error("Error saving screenshot", e)

    def spawn_debug_window(self):
        cv2.namedWindow(cfg.debug_window_name)
        if cfg.debug_window_always_on_top:
            hwnd = win32gui.FindWindow(None, cfg.debug_window_name)
            # Calculate center position of the screen
            screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            center_x = (screen_width - cfg.detection_window_width) // 2
            center_y = (screen_height - cfg.detection_window_height) // 2
            
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 
                                 center_x, center_y, 
                                 cfg.detection_window_width, cfg.detection_window_height, 0)

    def annotate_with_supervision(self, image, detections):
        if len(detections.xyxy) == 0:
            return image
        annotated_image = self.bounding_box_annotator.annotate(scene=image, detections=detections)
        if cfg.show_labels or cfg.show_conf or cfg.show_id:
            labels = []
            for conf, tracker_id in zip(detections.confidence, detections.tracker_id or [None]*len(detections)):
                parts = ["player"] if cfg.show_labels else []
                if cfg.show_conf:
                    parts.append(f"{conf:.2f}")
                if cfg.show_id and tracker_id is not None:
                    parts.append(f"ID: {int(tracker_id)}")
                labels.append(" ".join(parts))
            annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image

    def cleanup(self):
        self.running = False
        if cfg.show_window:
            cv2.destroyAllWindows()
        if cfg.show_overlay and hasattr(self, 'root'):
            self.root.destroy()
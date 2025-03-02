import sys
import subprocess
import time
import os
import threading
import queue
import cv2
import bettercam
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
import win32api
import cProfile
import pstats
import atexit
from config import cfg
from visualization import Visualization
from input import InputWatcher
from control import Controller
from logger import setup_logging, log_error

MAX_DETECTIONS = 100

class Target:
    def __init__(self, x1, y1, x2, y2, cls, id=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = x2 - x1
        self.h = y2 - y1
        self.cls = cls
        self.id = id
        self.center_x = (x1 + x2) / 2
        self.center_y = (y1 + y2) / 2
        self.target_x = self.center_x
        self.target_y = self.center_y + self.h * (cfg.body_y_offset - 0.5)

class UnifiedApp:
    def __init__(self, profile_duration=None):
        setup_logging()
        log_error("Starting UnifiedApp initialization")
        self.profile_duration = profile_duration
        if profile_duration:
            self.profile = cProfile.Profile()
            self.profile.enable()
            atexit.register(self.save_profile)
            self.start_time = time.time()
            self.frame_count = 0

        # Capture initialization
        self.daemon = True
        self.frame_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        self.screen_x_center = self.screen_width // 2
        self.screen_y_center = self.screen_height // 2
        self.prev_dims = (cfg.detection_window_width, cfg.detection_window_height, cfg.bettercam_capture_fps)
        self.bc = None
        self.setup_bettercam()

        # Detection initialization
        self.model = YOLO(f"models/{cfg.ai_model_name}", task="detect")
        self.tracker = sv.ByteTrack() if not cfg.ai_disable_tracker else None
        self.current_locked_target_id = None
        self.previous_target = None
        self.switch_threshold = cfg.switch_threshold
        self.detection_width = cfg.detection_window_width
        self.detection_height = cfg.detection_window_height
        self.center_np = np.array([self.detection_width / 2, self.detection_height / 2], dtype=np.float32)
        self.boxes_array = np.zeros((MAX_DETECTIONS, 4), dtype=np.float32)
        self.ids_array = np.zeros(MAX_DETECTIONS, dtype=np.int32)
        self.confidence_array = np.zeros(MAX_DETECTIONS, dtype=np.float32)
        self.cx_array = np.zeros(MAX_DETECTIONS, dtype=np.float32)
        self.cy_array = np.zeros(MAX_DETECTIONS, dtype=np.float32)
        self.distance_sq_array = np.zeros(MAX_DETECTIONS, dtype=np.float32)
        self.detections = sv.Detections.empty()

        # App components
        self.visualization = Visualization()
        self.input = InputWatcher(self)
        self.controller = Controller()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

        # Utils integration
        self._torch_device_cache = None
        log_error("UnifiedApp initialization completed")

    def setup_bettercam(self):
        region = (self.screen_x_center - cfg.detection_window_width // 2,
                  self.screen_y_center - cfg.detection_window_height // 2,
                  self.screen_x_center + cfg.detection_window_width // 2,
                  self.screen_y_center + cfg.detection_window_height // 2)
        self.bc = bettercam.create(
            device_idx=cfg.bettercam_monitor_id,
            output_idx=cfg.bettercam_gpu_id,
            output_color="BGR",
            max_buffer_len=16,
            region=region
        )
        if not self.bc.is_capturing:
            self.bc.start(region=region, target_fps=cfg.bettercam_capture_fps)

    def capture_frame(self):
        frame = self.bc.get_latest_frame()
        if frame is not None and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def capture_loop(self):
        while not self._stop_event.is_set():
            frame = self.capture_frame()
            if frame is not None:
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
            time.sleep(0.001)

    def get_new_frame(self):
        try:
            return self.frame_queue.get(timeout=0.17)
        except queue.Empty:
            return None

    def restart_capture(self):
        if (self.prev_dims[0] != cfg.detection_window_width or 
            self.prev_dims[1] != cfg.detection_window_height or 
            self.prev_dims[2] != cfg.bettercam_capture_fps):
            self.bc.stop()
            self.setup_bettercam()
            self.prev_dims = (cfg.detection_window_width, cfg.detection_window_height, cfg.bettercam_capture_fps)
            log_error("Capture reloaded")

    @torch.inference_mode()
    def perform_detection(self, image):
        results = self.model.predict(
            source=[np.ascontiguousarray(image)],
            imgsz=cfg.ai_model_image_size,
            stream=True,
            stream_buffer=True,
            nms=False,
            conf=cfg.ai_conf,
            iou=0.7,
            device=self.get_torch_device(),
            half=True,
            max_det=MAX_DETECTIONS,
            verbose=False
        )
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            return self.tracker.update_with_detections(detections) if self.tracker else detections
        return sv.Detections.empty()

    def sort_targets(self, frame):
        if not frame.xyxy.size:
            return None

        num_detections = min(len(frame.xyxy), MAX_DETECTIONS)
        self.boxes_array[:num_detections] = frame.xyxy[:num_detections]
        self.ids_array[:num_detections] = frame.tracker_id[:num_detections] if frame.tracker_id is not None else np.zeros(num_detections, dtype=np.int32)
        self.confidence_array[:num_detections] = frame.confidence[:num_detections] if frame.confidence is not None else np.ones(num_detections, dtype=np.float32)

        if num_detections == 0:
            return None

        if self.current_locked_target_id is not None:
            locked_idx = self._get_locked_index(self.ids_array[:num_detections])
            if locked_idx != -1:
                box = frame.xyxy[locked_idx]
                target = Target(*box, 0, self.ids_array[locked_idx])
                current_pos = np.array([target.center_x, target.center_y])
                prev_pos = np.array([self.previous_target.center_x, self.previous_target.center_y]) if self.previous_target else current_pos
                if np.linalg.norm(current_pos - prev_pos) <= self.switch_threshold:
                    return target

        return self._find_nearest_target_np(self.boxes_array[:num_detections], 
                                            self.ids_array[:num_detections], 
                                            self.confidence_array[:num_detections], 
                                            num_detections)

    def _find_nearest_target_np(self, boxes_array, ids_array, confidence_array, num_detections):
        # Pre-allocated NumPy arrays for calculations
        self.cx_array[:num_detections] = (boxes_array[:num_detections, 0] + boxes_array[:num_detections, 2]) / 2
        self.cy_array[:num_detections] = (boxes_array[:num_detections, 1] + boxes_array[:num_detections, 3]) / 2
        dx = self.cx_array[:num_detections] - self.center_np[0]
        dy = self.cy_array[:num_detections] - self.center_np[1]
        self.distance_sq_array[:num_detections] = dx * dx + dy * dy / (confidence_array[:num_detections] + 1e-6)

        best_idx = np.argmin(self.distance_sq_array[:num_detections])
        if self.distance_sq_array[best_idx] == float('inf'):
            return None

        target_info = (boxes_array[best_idx, 0], boxes_array[best_idx, 1], 
                       boxes_array[best_idx, 2], boxes_array[best_idx, 3], 
                       0, ids_array[best_idx])
        return Target(*target_info)

    def _get_locked_index(self, tracker_ids):
        try:
            return int(np.where(tracker_ids == self.current_locked_target_id)[0][0])
        except (IndexError, ValueError):
            return -1

    def handle_target(self, target):
        if not target:
            self.current_locked_target_id = None
            self.previous_target = None
            return

        if self.current_locked_target_id is None or self.current_locked_target_id != target.id:
            self.current_locked_target_id = target.id

        shooting_state = self.controller.get_shooting_state()
        self.controller.process_target(target.target_x, target.target_y, target.w, target.h, shooting_state)
        self.previous_target = target

    def run(self):
        while True:
            try:
                if self.profile_duration and time.time() - self.start_time > self.profile_duration:
                    print(f"Profiling complete after {self.profile_duration} seconds, {self.frame_count} frames")
                    self.cleanup()
                    sys.exit(0)

                if self.input.app_pause:
                    time.sleep(0.1)
                    continue

                image = self.get_new_frame()
                if image is None:
                    time.sleep(1 / cfg.bettercam_capture_fps)
                    continue

                self.detections = self.perform_detection(image)
                target = self.sort_targets(self.detections)
                self.handle_target(target)

                if cfg.show_window or cfg.show_overlay:
                    self.visualization.queue.put((image, self.detections))

                if self.profile_duration:
                    self.frame_count += 1

            except Exception as e:
                log_error("Error in main loop", e)
                time.sleep(0.1)

    def cleanup(self):
        self._stop_event.set()
        if self.bc and self.bc.is_capturing:
            self.bc.stop()
        self.capture_thread.join()
        try:
            if self.visualization and self.visualization.running:
                self.visualization.queue.put(None)
                time.sleep(0.1)
                self.visualization.cleanup()
            cv2.destroyAllWindows()
            torch.cuda.empty_cache()
        except Exception as e:
            log_error(f"Error during cleanup: {e}")
        if self.input:
            self.input.quit()
        if self.profile_duration:
            self.save_profile()
        log_error("UnifiedApp cleaned up")

    def restart(self):
        print("Restarting app...")
        self.cleanup()
        time.sleep(0.5)
        python = sys.executable
        args = sys.argv[:]
        subprocess.Popen([python] + args)
        os._exit(0)

    def save_profile(self):
        if not self.profile_duration:
            return
        self.profile.disable()
        print("Saving profiling data...")
        self.profile.dump_stats('unified_app_profile.prof')
        with open('unified_app_profile_report.txt', 'w') as f:
            stats = pstats.Stats(self.profile, stream=f).sort_stats('cumulative')
            stats.print_stats(50)
        print("Profiling data saved to unified_app_profile.prof")
        print("Human-readable report saved to unified_app_profile_report.txt")

    def get_torch_device(self):
        if self._torch_device_cache is None:
            ai_device = str(cfg.ai_device).strip().lower()
            if cfg.ai_enable_amd:
                self._torch_device_cache = torch.device(f'hip:{ai_device}')
            elif 'cpu' in ai_device:
                self._torch_device_cache = torch.device('cpu')
            elif ai_device.isdigit():
                self._torch_device_cache = torch.device(f'cuda:{ai_device}')
            else:
                self._torch_device_cache = torch.device('cuda:0')
        return self._torch_device_cache

def main():
    profile_duration = None  # Set to 30 or another value for profiling
    app = UnifiedApp(profile_duration=profile_duration)
    cfg.set_restart_callback(app.restart)
    try:
        app.run()
    except KeyboardInterrupt:
        app.cleanup()
    os._exit(0)

if __name__ == "__main__":
    main()
import configparser
import os
from threading import Lock
from gui import GUI
from logger import log_error

class Config:
    CONFIG_SECTIONS = {
        "DETECTION_WINDOW": "Detection window",
        "AI": "AI",
        "MOUSE": "Mouse",
        "AIM": "Aim",
        "TRIGGERBOT": "Triggerbot",
        "ARDUINO": "Arduino",
        "DEBUG_WINDOW": "Debug window",
        "OVERLAY": "overlay",
        "CAPTURE_METHODS": "Capture Methods",
        "HOTKEYS": "Hotkeys",
        "MLP": "Neural Network",
        "MODEL_EXPORT": "Model Export"
    }

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.editor_lock = Lock()
        self.restart_callback = None
        self.read(verbose=False)

    def set_restart_callback(self, callback):
        self.restart_callback = callback

    def edit_config(self):
        with self.editor_lock:
            try:
                editor = GUI(self, self.restart_callback)
                editor.show()
            except Exception as e:
                log_error(f"Error opening config editor: {e}")

    def read(self, verbose=True):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
        if not os.path.isfile(config_path):
            self.write()
        self.config.read(config_path)
        if verbose:
            print("Config reloaded")

    def write(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
        with open(config_path, 'w') as configfile:
            self.config.write(configfile)

    @property
    def detection_window_width(self):
        return int(self.config[self.CONFIG_SECTIONS["DETECTION_WINDOW"]]["detection_window_width"])

    @property
    def detection_window_height(self):
        return int(self.config[self.CONFIG_SECTIONS["DETECTION_WINDOW"]]["detection_window_height"])

    @property
    def use_padding(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DETECTION_WINDOW"], "use_padding")

    @property
    def shared_memory_usage(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DETECTION_WINDOW"], "shared_memory_usage")

    @property
    def show_trajectory(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DETECTION_WINDOW"], "show_trajectory")

    @property
    def ai_model_name(self):
        return self.config[self.CONFIG_SECTIONS["AI"]]["ai_model_name"]

    @property
    def ai_model_image_size(self):
        return int(self.config[self.CONFIG_SECTIONS["AI"]]["ai_model_image_size"])

    @property
    def ai_conf(self):
        return float(self.config[self.CONFIG_SECTIONS["AI"]]["ai_conf"])

    @property
    def ai_device(self):
        return self.config[self.CONFIG_SECTIONS["AI"]]["ai_device"]

    @property
    def ai_disable_tracker(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["AI"], "ai_disable_tracker")

    @property
    def ai_enable_amd(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["AI"], "ai_enable_amd")

    @property
    def mouse_speed(self):
        return int(self.config[self.CONFIG_SECTIONS["MOUSE"]]["mouse_speed"])

    @property
    def mouse_speed_boost(self):
        return float(self.config[self.CONFIG_SECTIONS["MOUSE"]]["mouse_speed_boost"])

    @property
    def mouse_boost_threshold(self):
        return float(self.config[self.CONFIG_SECTIONS["MOUSE"]]["mouse_boost_threshold"])

    @property
    def mouse_lock_target(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MOUSE"], "mouse_lock_target")

    @property
    def mouse_auto_aim(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MOUSE"], "mouse_auto_aim")

    @property
    def mouse_ghub(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MOUSE"], "mouse_ghub")

    @property
    def mouse_rzr(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MOUSE"], "mouse_rzr")

    @property
    def body_y_offset(self):
        return float(self.config[self.CONFIG_SECTIONS["AIM"]]["body_y_offset"])

    @property
    def hideout_targets(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["AIM"], "hideout_targets")

    @property
    def disable_headshot(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["AIM"], "disable_headshot")

    @property
    def disable_prediction(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["AIM"], "disable_prediction")

    @property
    def prediction_interval(self):
        return float(self.config[self.CONFIG_SECTIONS["AIM"]]["prediction_interval"])

    @property
    def third_person(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["AIM"], "third_person")

    @property
    def switch_threshold(self):
        return float(self.config[self.CONFIG_SECTIONS["AIM"]]["switch_threshold"])

    @property
    def smoothing_factor(self):
        return float(self.config[self.CONFIG_SECTIONS["AIM"]]["smoothing_factor"])

    @property
    def prioritize_headshot(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["AIM"], "prioritize_headshot")

    @property
    def triggerbot_hotkey(self):
        return self.config[self.CONFIG_SECTIONS["TRIGGERBOT"]]["triggerbot_hotkey"]

    @property
    def triggerbot(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["TRIGGERBOT"], "triggerbot")

    @property
    def force_click(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["TRIGGERBOT"], "force_click")

    @property
    def tbot_box_size(self):
        return float(self.config[self.CONFIG_SECTIONS["TRIGGERBOT"]]["triggerbot_box_size"])

    @property
    def arduino_move(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["ARDUINO"], "arduino_move")

    @property
    def arduino_shoot(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["ARDUINO"], "arduino_shoot")

    @property
    def arduino_port(self):
        return self.config[self.CONFIG_SECTIONS["ARDUINO"]]["arduino_port"]

    @property
    def arduino_baudrate(self):
        return int(self.config[self.CONFIG_SECTIONS["ARDUINO"]]["arduino_baudrate"])

    @property
    def arduino_16_bit_mouse(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["ARDUINO"], "arduino_16_bit_mouse")

    @property
    def show_window(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_window")

    @property
    def show_detection_speed(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_detection_speed")

    @property
    def show_window_fps(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_window_fps")

    @property
    def show_boxes(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_boxes")

    @property
    def show_labels(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_labels")

    @property
    def show_conf(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_conf")

    @property
    def show_id(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_id")

    @property
    def show_target_line(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_target_line")

    @property
    def show_target_prediction_line(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_target_prediction_line")

    @property
    def show_triggerbot_box(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_triggerbot_box")

    @property
    def show_history_points(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "show_history_points")

    @property
    def debug_window_always_on_top(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["DEBUG_WINDOW"], "debug_window_always_on_top")

    @property
    def spawn_window_pos_x(self):
        return int(self.config[self.CONFIG_SECTIONS["DEBUG_WINDOW"]]["spawn_window_pos_x"])

    @property
    def spawn_window_pos_y(self):
        return int(self.config[self.CONFIG_SECTIONS["DEBUG_WINDOW"]]["spawn_window_pos_y"])

    @property
    def debug_window_scale_percent(self):
        return int(self.config[self.CONFIG_SECTIONS["DEBUG_WINDOW"]]["debug_window_scale_percent"])

    @property
    def debug_window_screenshot_key(self):
        return self.config[self.CONFIG_SECTIONS["DEBUG_WINDOW"]]["debug_window_screenshot_key"]

    @property
    def debug_window_name(self):
        return 'OBS STREAM'  # Fixed value as per original

    @property
    def show_overlay(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["OVERLAY"], "show_overlay")

    @property
    def overlay_show_borders(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["OVERLAY"], "overlay_show_borders")

    @property
    def overlay_show_boxes(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["OVERLAY"], "overlay_show_boxes")

    @property
    def overlay_show_target_line(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["OVERLAY"], "overlay_show_target_line")

    @property
    def overlay_show_target_prediction_line(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["OVERLAY"], "overlay_show_target_prediction_line")

    @property
    def overlay_show_labels(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["OVERLAY"], "overlay_show_labels")

    @property
    def overlay_show_conf(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["OVERLAY"], "overlay_show_conf")

    @property
    def mss_capture(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["CAPTURE_METHODS"], "mss_capture")

    @property
    def mss_capture_fps(self):
        return int(self.config[self.CONFIG_SECTIONS["CAPTURE_METHODS"]]["mss_fps"])

    @property
    def bettercam_capture(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["CAPTURE_METHODS"], "bettercam_capture")

    @property
    def bettercam_capture_fps(self):
        return int(self.config[self.CONFIG_SECTIONS["CAPTURE_METHODS"]]["bettercam_capture_fps"])

    @property
    def bettercam_monitor_id(self):
        return int(self.config[self.CONFIG_SECTIONS["CAPTURE_METHODS"]]["bettercam_monitor_id"])

    @property
    def bettercam_gpu_id(self):
        return int(self.config[self.CONFIG_SECTIONS["CAPTURE_METHODS"]]["bettercam_gpu_id"])

    @property
    def obs_capture(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["CAPTURE_METHODS"], "obs_capture")

    @property
    def obs_camera_id(self):
        return self.config[self.CONFIG_SECTIONS["CAPTURE_METHODS"]]["obs_camera_id"]

    @property
    def obs_capture_fps(self):
        return int(self.config[self.CONFIG_SECTIONS["CAPTURE_METHODS"]]["obs_capture_fps"])

    @property
    def hotkey_targeting(self):
        return self.config[self.CONFIG_SECTIONS["HOTKEYS"]]["hotkey_targeting"].split(",")

    @property
    def hotkey_exit(self):
        return self.config[self.CONFIG_SECTIONS["HOTKEYS"]]["hotkey_exit"]

    @property
    def hotkey_pause(self):
        return self.config[self.CONFIG_SECTIONS["HOTKEYS"]]["hotkey_pause"]

    @property
    def hotkey_reload_config(self):
        return self.config[self.CONFIG_SECTIONS["HOTKEYS"]]["hotkey_reload_config"]

    @property
    def hotkey_edit_config(self):
        return self.config[self.CONFIG_SECTIONS["HOTKEYS"]]["hotkey_edit_config"]

    @property
    def hotkey_triggerbot(self):
        return self.config[self.CONFIG_SECTIONS["HOTKEYS"]]["hotkey_triggerbot"]

    @property
    def mouse_mlp(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MLP"], "mouse_mlp")

    @property
    def mlp_samples(self):
        return int(self.config[self.CONFIG_SECTIONS["MLP"]]["mlp_samples"])

    @property
    def mlp_epochs(self):
        return int(self.config[self.CONFIG_SECTIONS["MLP"]]["mlp_epochs"])

    @property
    def batch_size(self):
        return int(self.config[self.CONFIG_SECTIONS["MLP"]]["batch_size"])

    @property
    def warmup_epochs(self):
        return int(self.config[self.CONFIG_SECTIONS["MLP"]]["warmup_epochs"])

    @property
    def min_lr(self):
        return float(self.config[self.CONFIG_SECTIONS["MLP"]]["min_lr"])

    @property
    def initial_lr(self):
        return float(self.config[self.CONFIG_SECTIONS["MLP"]]["initial_lr"])

    @property
    def lr_factor(self):
        return float(self.config[self.CONFIG_SECTIONS["MLP"]]["lr_factor"])

    @property
    def lr_patience(self):
        return int(self.config[self.CONFIG_SECTIONS["MLP"]]["lr_patience"])

    @property
    def checkpoint_freq(self):
        return int(self.config[self.CONFIG_SECTIONS["MLP"]]["checkpoint_freq"])

    @property
    def patience(self):
        return int(self.config[self.CONFIG_SECTIONS["MLP"]]["patience"])

    @property
    def load_existing_mlp(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MLP"], "load_existing_mlp")

    @property
    def max_target_speed(self):
        return int(self.config[self.CONFIG_SECTIONS["MLP"]]["max_target_speed"])
    # [Model Export] Properties
    @property
    def export_model(self):
        return self.config[self.CONFIG_SECTIONS["MODEL_EXPORT"]]["model"]

    @property
    def export_format(self):
        return self.config[self.CONFIG_SECTIONS["MODEL_EXPORT"]]["format"]

    @property
    def export_imgsz(self):
        return int(self.config[self.CONFIG_SECTIONS["MODEL_EXPORT"]]["imgsz"])

    @property
    def export_half(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MODEL_EXPORT"], "half")

    @property
    def export_int8(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MODEL_EXPORT"], "int8")

    @property
    def export_dynamic(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MODEL_EXPORT"], "dynamic")

    @property
    def export_simplify(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MODEL_EXPORT"], "simplify")

    @property
    def export_nms(self):
        return self.config.getboolean(self.CONFIG_SECTIONS["MODEL_EXPORT"], "nms")

    @property
    def export_batch(self):
        return int(self.config[self.CONFIG_SECTIONS["MODEL_EXPORT"]]["batch"])

    @property
    def export_device(self):
        return int(self.config[self.CONFIG_SECTIONS["MODEL_EXPORT"]]["device"])

    @property
    def export_iou(self):
        return float(self.config["Model Export"]["iou"])
    @property
    def export_conf(self):
        return float(self.config["Model Export"]["conf"])
cfg = Config()
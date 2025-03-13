import tkinter as tk
import time
import os
import subprocess
from tkinter import ttk
from PIL import Image, ImageTk
from logger import log_error

class GUI:
    def __init__(self, config_obj, restart_callback=None):
        self.root = tk.Tk()
        self.root.title("Settings")
        self.restart_callback = restart_callback
        self.config = config_obj.config
        png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extras", "projectalpha.png")
        try:
            self.bg_image = Image.open(png_path)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        except Exception as e:
            log_error(f"Error loading PNG: {e}")
            self.bg_photo = None
        try:
            config_obj.read(verbose=True)
        except Exception as e:
            log_error(f"Error loading config: {e}")
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both')
        style = ttk.Style()
        style.theme_use('default')
        font_config = ("Helvetica", 16)
        style.configure("Custom.TNotebook.Tab", background="purple1", foreground="white")
        style.map("Custom.TNotebook.Tab", 
                  background=[("active", "thistle1"), ("!active", "dark orchid")],
                  foreground=[("active", "black"), ("!active", "white")])
        style.configure("Custom.TButton", background="purple1", foreground="white")
        style.map("Custom.TButton", 
                  background=[("active", "thistle1"), ("!active", "DarkOrchid1")],
                  foreground=[("active", "black"), ("!active", "white")])
        style.configure("Custom.TLabel", background="#ADD8E6", foreground="black", font=font_config)
        style.configure("Custom.TCombobox", fieldbackground="#ADD8E6", background="#ADD8E6", foreground="black", font=font_config)
        style.configure("Custom.TEntry", fieldbackground="#ADD8E6", foreground="black", font=font_config)
        self.notebook = ttk.Notebook(main_frame, style="Custom.TNotebook")
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        self.widgets = {}
        self.variables = {}
        self.canvases = {}
        
        current_config = self.config
        
        for section in config_obj.CONFIG_SECTIONS.values():
            tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(tab_frame, text=section)
            canvas = tk.Canvas(tab_frame)
            canvas.pack(expand=True, fill='both')
            if self.bg_photo:
                canvas.create_image(0, 0, anchor='nw', image=self.bg_photo, tags="bg")
                canvas.lower("bg")
            self.canvases[section] = canvas
            canvas.bind('<Configure>', lambda e, s=section: self._resize_background(s, e.width, e.height))
            section_items = current_config[section]
            row = 0
            for key, value in section_items.items():
                label = ttk.Label(canvas, text=key, style="Custom.TLabel")
                label.grid(row=row, column=0, padx=5, pady=5)             
                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    widget = ttk.Combobox(canvas, values=['True', 'False'], state='readonly', style="Custom.TCombobox")
                    widget.set(str(value))
                    widget.bind('<<ComboboxSelected>>', lambda e, v=var: v.set(e.widget.get() == 'True'))
                else:
                    var = tk.StringVar(value=str(value))
                    widget = ttk.Entry(canvas, style="Custom.TEntry", textvariable=var)
                widget.grid(row=row, column=1, padx=5, pady=5)
                self.variables[(section, key)] = var
                self.widgets[(section, key)] = widget
                row += 1
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side='bottom', fill='x', pady=10)
        save_btn = ttk.Button(button_frame, text="Save", command=self.save_config, style="Custom.TButton")
        save_btn.pack(side='left', padx=5)
        run_restart_btn = ttk.Button(button_frame, text="Run/Restart", command=self.run_restart, style="Custom.TButton")
        run_restart_btn.pack(side='left', padx=5)
        start_bot_btn = ttk.Button(button_frame, text="Start Bot", command=self.start_bot, style="Custom.TButton")
        start_bot_btn.pack(side='left', padx=5)
        train_mlp_btn = ttk.Button(button_frame, text="Train MLP", command=self.train_mlp, style="Custom.TButton")
        train_mlp_btn.pack(side='left', padx=5)
        export_model_btn = ttk.Button(button_frame, text="Export Model", command=self.export_model, style="Custom.TButton")
        export_model_btn.pack(side='left', padx=5)
        close_btn = ttk.Button(button_frame, text="Close", command=self.close, style="Custom.TButton")
        close_btn.pack(side='left', padx=5)
        # YOLO buttons
        yolo_frame = ttk.Frame(main_frame)
        yolo_frame.place(relx=0.85, rely=0.95, anchor='se')
        yolo_new_btn = ttk.Button(yolo_frame, text="YOLO_new", command=self.run_yolo_new, style="Custom.TButton")
        yolo_new_btn.pack(side='left', padx=5, pady=5)
        yolo_finetune_btn = ttk.Button(yolo_frame, text="Yolo_finetune", command=self.run_yolo_finetune, style="Custom.TButton")
        yolo_finetune_btn.pack(side='left', padx=5, pady=5)
        annotate_btn = ttk.Button(yolo_frame, text="Annotate", command=self.run_annotate, style="Custom.TButton")
        annotate_btn.pack(side='left', padx=5, pady=5)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _resize_background(self, section, width, height):
        if not self.bg_photo or width <= 1 or height <= 1:
            return
        canvas = self.canvases[section]
        original_width, original_height = self.bg_image.size
        width_ratio = width / original_width
        height_ratio = height / original_height
        scale_ratio = min(width_ratio, height_ratio)
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        x_position = (width - new_width) // 2
        y_position = (height - new_height) // 2
        resized_image = self.bg_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_photo = ImageTk.PhotoImage(resized_image)
        canvas.bg_image = resized_photo
        canvas.delete("bg")
        canvas.create_image(x_position, y_position, anchor='nw', image=resized_photo, tags="bg")
        canvas.lower("bg")

    def save_config(self):
        for (section, key), var in self.variables.items():
            value = var.get()
            self.config[section][key] = str(value)
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "config.ini")
        with open(config_path, 'w') as configfile:
            self.config.write(configfile)

    def run_restart(self):
        self.save_config()
        if self.restart_callback:
            try:
                self.close()
                time.sleep(0.1)
                self.restart_callback()
                os._exit(0)
            except Exception as e:
                log_error(f"Error during run/restart: {e}")
                os._exit(1)
        else:
            self.start_bot()

    def start_bot(self):
        try:
            directory = os.path.dirname(os.path.abspath(__file__))
            run_script = os.path.join(directory, "app.py")
            if not os.path.exists(run_script):
                log_error(f"app.py not found at {run_script}")
                return
            subprocess.Popen(["python", run_script], cwd=directory)
            print("Started app.py")
        except Exception as e:
            log_error(f"Error starting app.py: {e}")

    def export_model(self):
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(__file__), "models", cfg.export_model)
            if not os.path.exists(model_path):
                log_error(f"Model file not found: {model_path}")
                return
            
            model = YOLO(model_path)
            export_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(export_dir, exist_ok=True)
            exported_path = model.export(
                format=cfg.export_format,
                imgsz=cfg.export_imgsz,
                half=cfg.export_half,
                int8=cfg.export_int8,
                dynamic=cfg.export_dynamic,
                simplify=cfg.export_simplify,
                nms=cfg.export_nms,
                batch=cfg.export_batch,
                device=cfg.export_device,
                iou=cfg.export_iou,
                conf=cfg.export_conf
            )
            log_error(f"Model exported successfully to {exported_path}")
            print(f"Model exported to {exported_path}")
        except Exception as e:
            log_error(f"Error exporting model: {e}")

    def train_mlp(self):
        try:
            directory = os.path.dirname(os.path.abspath(__file__))
            train_script = os.path.join(directory, "train.py")
            if not os.path.exists(train_script):
                log_error(f"train.py not found at {train_script}")
                return
            subprocess.Popen(["python", train_script], cwd=directory)
            print("Started train.py")
        except Exception as e:
            log_error(f"Error starting train.py: {e}")

    def run_yolo_new(self):
        try:
            directory = os.path.dirname(os.path.abspath(__file__))
            train_script = os.path.join(directory, "train_yolo_new.py")
            if not os.path.exists(train_script):
                log_error(f"train_yolo_new.py not found at {train_script}") 
                return
            subprocess.Popen(["python", train_script], cwd=directory)
            print("Started train_yolo_new.py")
        except Exception as e:
            log_error(f"Error starting train_yolo_new.py: {e}")

    def run_yolo_finetune(self):
        try:
            directory = os.path.dirname(os.path.abspath(__file__))
            finetune_script = os.path.join(directory, "finetune_yolo_existing.py")
            if not os.path.exists(finetune_script):
                log_error(f"finetune_yolo_existing.py not found at {finetune_script}")
                return
            subprocess.Popen(["python", finetune_script], cwd=directory)
            print("Started finetune_yolo_existing.py")
        except Exception as e:
            log_error(f"Error starting finetune_yolo_existing.py: {e}")

    def run_annotate(self):
        try:
            directory = os.path.dirname(os.path.abspath(__file__))
            annotate_script = os.path.join(directory, "annotate_dataset.py")
            if not os.path.exists(annotate_script):
                log_error(f"annotate_dataset.py not found at {annotate_script}")
                return
            subprocess.Popen(["python", annotate_script], cwd=directory)
            print("Started annotate_dataset.py")
        except Exception as e:
            log_error(f"Error starting annotate_dataset.py: {e}")

    def close(self):
        if self.root and self.root.winfo_exists():
            self.root.destroy()
            self.root.quit()

    def show(self):
        self.root.mainloop()

if __name__ == "__main__":
    from config import cfg
    editor = GUI(cfg)
    editor.show()
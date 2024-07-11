import tkinter as tk
from tkinter import filedialog, ttk
import subprocess
import os
import threading

def run_prepare_data(data_dir, status_var, progress_bar):
    try:
        status_var.set("Preparing data...")
        progress_bar.start()
        script_path = "src/main/python/data/prepare_data.py"
        subprocess.run(["python", script_path, data_dir], check=True)
        status_var.set("Data preparation completed")
    except subprocess.CalledProcessError as e:
        status_var.set(f"Error: {e}")
    finally:
        progress_bar.stop()

def run_train_model(data_dir, model_save_path, status_var, progress_bar):
    try:
        status_var.set("Training model...")
        progress_bar.start()
        script_path = "src/main/python/train.py"
        subprocess.run(["python", script_path, data_dir, model_save_path], check=True)
        status_var.set("Training completed")
    except subprocess.CalledProcessError as e:
        status_var.set(f"Error: {e}")
    finally:
        progress_bar.stop()

def run_infer(model_dir, video_file, status_var, progress_bar):
    try:
        status_var.set("Running inference...")
        progress_bar.start()
        script_path = "src/main/python/infer.py"
        subprocess.run(["python", script_path, model_dir, video_file], check=True)
        status_var.set("Inference completed")
    except subprocess.CalledProcessError as e:
        status_var.set(f"Error: {e}")
    finally:
        progress_bar.stop()

def select_prepare_data_directory(status_var, progress_bar):
    data_dir = filedialog.askdirectory(title="Select Directory with Data to Prepare")
    if data_dir:
        threading.Thread(target=run_prepare_data, args=(data_dir, status_var, progress_bar)).start()

def select_train_model_directories(status_var, progress_bar):
    data_dir = filedialog.askdirectory(title="Select Directory with Prepared Data")
    if not data_dir:
        status_var.set("No directory selected. Training cancelled.")
        return
    model_save_path = filedialog.asksaveasfilename(title="Save Model As", defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
    if not model_save_path:
        status_var.set("No save path selected. Training cancelled.")
        return
    threading.Thread(target=run_train_model, args=(data_dir, model_save_path, status_var, progress_bar)).start()

def select_infer_directories(status_var, progress_bar):
    model_dir = filedialog.askdirectory(title="Select Directory with Model")
    if not model_dir:
        status_var.set("No directory selected. Inference cancelled.")
        return
    video_file = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_file:
        status_var.set("No video file selected. Inference cancelled.")
        return
    threading.Thread(target=run_infer, args=(model_dir, video_file, status_var, progress_bar)).start()

def main():
    root = tk.Tk()
    root.title("AI Object Finder")

    frame = tk.Frame(root)
    frame.pack(pady=20)

    label = tk.Label(frame, text="AI Object Finder", font=("Helvetica", 16))
    label.pack(pady=10)

    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)

    status_var = tk.StringVar()
    status_var.set("Idle")

    status_label = tk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor="w")
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    progress_bar = ttk.Progressbar(root, mode='indeterminate')
    progress_bar.pack(side=tk.BOTTOM, fill=tk.X)

    prepare_data_button = tk.Button(button_frame, text="Prepare Data", command=lambda: select_prepare_data_directory(status_var, progress_bar))
    prepare_data_button.pack(side=tk.LEFT, padx=5)

    train_model_button = tk.Button(button_frame, text="Train Model", command=lambda: select_train_model_directories(status_var, progress_bar))
    train_model_button.pack(side=tk.LEFT, padx=5)

    infer_button = tk.Button(button_frame, text="Infer", command=lambda: select_infer_directories(status_var, progress_bar))
    infer_button.pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import threading
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.datasets import VideoDataset
from models.i3d import I3D
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.prepare_data import prepare_data
from train import main as train_main
from infer import main as infer_main
from models.efficientdet_3d import get_efficientdet_3d

class LogRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        self.widget.insert(tk.END, message)
        self.widget.see(tk.END)

    def flush(self):
        pass

def run_prepare_data(labels_file, output_dir, base_video_dir, status_var, progress_bar, log_text):
    try:
        status_var.set("Preparing data...")
        progress_bar.start()
        
        log_callback = lambda message: log_text.insert(tk.END, message + "\n")
        prepare_data(labels_file, output_dir, base_video_dir, log_callback)
        
        status_var.set("Data preparation completed")
        progress_bar.stop()
    except Exception as e:
        status_var.set(f"Error: {e}")
        progress_bar.stop()


def run_train_model(data_dir, model_save_path, model_type, status_var, progress_bar, log_text):
    try:
        status_var.set("Training model...")
        progress_bar.start()

        # Redirect stdout to log_text
        sys.stdout = LogRedirector(log_text)

        log_callback = lambda message: log_text.insert(tk.END, message + "\n")
        
        if model_type == 'i3d':
            model = I3D().cuda()
        elif model_type == 'efficientdet':
            model = get_efficientdet_3d().cuda()
        else:
            raise ValueError("Invalid model type. Choose 'i3d' or 'efficientdet'.")
        
        train_main(data_dir, model_save_path, model_type, log_callback, model)

        status_var.set("Training completed")
        progress_bar.stop()

        # Reset stdout
        sys.stdout = sys.__stdout__

    except Exception as e:
        status_var.set(f"Error: {e}")
        progress_bar.stop()
        # Reset stdout
        sys.stdout = sys.__stdout__

def run_infer(model_path, test_csv, status_var, progress_bar, log_text):
    try:
        status_var.set("Running inference...")
        progress_bar.start()

        # Redirect stdout to log_text
        sys.stdout = LogRedirector(log_text)

        sys.argv = ['infer.py', model_path, test_csv]
        infer_main()

        status_var.set("Inference completed")
        progress_bar.stop()

        # Reset stdout
        sys.stdout = sys.__stdout__

    except Exception as e:
        status_var.set(f"Error: {e}")
        progress_bar.stop()
        # Reset stdout
        sys.stdout = sys.__stdout__

def select_prepare_data_files(status_var, progress_bar, log_text):
    labels_file = filedialog.askopenfilename(title="Select Labels File", filetypes=[("CSV Files", "*.csv")])
    if not labels_file:
        status_var.set("No labels file selected. Data preparation cancelled.")
        return
    output_dir = filedialog.askdirectory(title="Select Output Directory for Prepared Data")
    if not output_dir:
        status_var.set("No output directory selected. Data preparation cancelled.")
        return
    base_video_dir = filedialog.askdirectory(title="Select Base Directory for Video Files")
    if not base_video_dir:
        status_var.set("No base video directory selected. Data preparation cancelled.")
        return
    threading.Thread(target=run_prepare_data, args=(labels_file, output_dir, base_video_dir, status_var, progress_bar, log_text)).start()

def select_train_model_directories(status_var, progress_bar, log_text):
    data_dir = filedialog.askdirectory(title="Select Directory with Prepared Data")
    if not data_dir:
        status_var.set("No directory selected. Training cancelled.")
        return
    model_save_path = filedialog.asksaveasfilename(title="Save Model As", defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
    if not model_save_path:
        status_var.set("No save path selected. Training cancelled.")
        return
    model_type = select_model_type()
    if not model_type:
        status_var.set("No model type selected. Training cancelled.")
        return
    threading.Thread(target=run_train_model, args=(data_dir, model_save_path, model_type, status_var, progress_bar, log_text)).start()

def select_infer_directories(status_var, progress_bar, log_text):
    model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch Model", "*.pth")])
    if not model_path:
        status_var.set("No model file selected. Inference cancelled.")
        return
    test_csv = filedialog.askopenfilename(title="Select Test CSV File", filetypes=[("CSV Files", "*.csv")])
    if not test_csv:
        status_var.set("No test CSV file selected. Inference cancelled.")
        return
    threading.Thread(target=run_infer, args=(model_path, test_csv, status_var, progress_bar, log_text)).start()

def select_model_type():
    model_type = tk.StringVar()
    model_type.set("i3d")  # default value

    top = tk.Toplevel()
    top.title("Select Model Type")

    tk.Radiobutton(top, text="I3D", variable=model_type, value="i3d").pack()
    tk.Radiobutton(top, text="EfficientDet", variable=model_type, value="efficientdet").pack()

    def on_close():
        top.destroy()

    tk.Button(top, text="OK", command=on_close).pack()

    top.grab_set()
    top.wait_window()

    return model_type.get()

def main():
    root = tk.Tk()
    root.title("AI Theft Detection")

    frame = tk.Frame(root)
    frame.pack(pady=20)

    label = tk.Label(frame, text="AI Theft Detection", font=("Helvetica", 16))
    label.pack(pady=10)

    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)

    status_var = tk.StringVar()
    status_var.set("Idle")

    status_label = tk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor="w")
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    progress_bar = ttk.Progressbar(root, mode='indeterminate')
    progress_bar.pack(side=tk.BOTTOM, fill=tk.X)

    log_text = scrolledtext.ScrolledText(root, height=10)
    log_text.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    prepare_data_button = tk.Button(button_frame, text="Prepare Data", command=lambda: select_prepare_data_files(status_var, progress_bar, log_text))
    prepare_data_button.pack(side=tk.LEFT, padx=5)

    train_model_button = tk.Button(button_frame, text="Train Model", command=lambda: select_train_model_directories(status_var, progress_bar, log_text))
    train_model_button.pack(side=tk.LEFT, padx=5)

    infer_button = tk.Button(button_frame, text="Infer", command=lambda: select_infer_directories(status_var, progress_bar, log_text))
    infer_button.pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()
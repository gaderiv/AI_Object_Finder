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
from models.efficientdet import EfficientDet
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.prepare_data import prepare_data
from train import main as train_main

class LogRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        self.widget.insert(tk.END, message)
        self.widget.see(tk.END)

    def flush(self):
        pass

def run_prepare_data(labels_file, output_dir, status_var, progress_bar, log_text):
    try:
        status_var.set("Preparing data...")
        progress_bar.start()
        
        log_callback = lambda message: log_text.insert(tk.END, message + "\n")
        prepare_data(labels_file, output_dir, log_callback)
        
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
        train_main(data_dir, model_save_path, model_type, log_callback)

        status_var.set("Training completed")
        progress_bar.stop()

        # Reset stdout
        sys.stdout = sys.__stdout__

    except Exception as e:
        status_var.set(f"Error: {e}")
        progress_bar.stop()
        # Reset stdout
        sys.stdout = sys.__stdout__

def run_infer(model_path, video_file, model_type, status_var, progress_bar, log_text):
    try:
        status_var.set("Running inference...")
        progress_bar.start()

        # Redirect stdout to log_text
        sys.stdout = LogRedirector(log_text)

        from infer import load_model, infer_on_video, plot_confusion_matrix
        model = load_model(model_path, model_type)
        predictions = infer_on_video(model, video_file)

        # For demonstration, we're using a dummy true label. In a real scenario, you'd need to provide the actual label.
        true_label = [1]  # Replace with actual label
        threshold = 0.5
        predicted_label = (predictions > threshold).astype(int)

        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
        precision, recall, f1_score, _ = precision_recall_fscore_support(true_label, predicted_label, average='binary')
        conf_matrix = confusion_matrix(true_label, predicted_label)
        accuracy = accuracy_score(true_label, predicted_label)

        status_var.set(f"Inference completed. Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}, Accuracy: {accuracy:.2f}")
        progress_bar.stop()

        plot_confusion_matrix(conf_matrix)

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
    threading.Thread(target=run_prepare_data, args=(labels_file, output_dir, status_var, progress_bar, log_text)).start()

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
    video_file = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_file:
        status_var.set("No video file selected. Inference cancelled.")
        return
    model_type = select_model_type()
    if not model_type:
        status_var.set("No model type selected. Inference cancelled.")
        return
    threading.Thread(target=run_infer, args=(model_path, video_file, model_type, status_var, progress_bar, log_text)).start()

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
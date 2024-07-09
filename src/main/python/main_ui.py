import tkinter as tk
from tkinter import filedialog
import subprocess
import os

def run_script(script_path):
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script: {e}")

def select_script():
    script_path = filedialog.askopenfilename(title="Select Script", filetypes=[("Python Files", "*.py")])
    return script_path

def main():
    root = tk.Tk()
    root.title("AI Object Finder")

    frame = tk.Frame(root)
    frame.pack(pady=20)

    label = tk.Label(frame, text="AI Object Finder", font=("Helvetica", 16))
    label.pack(pady=10)

    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)

    prepare_data_button = tk.Button(button_frame, text="Prepare Data", command=lambda: run_script(os.path.join(os.getcwd(), "src", "main", "python", "data", "prepare_data.py")))
    prepare_data_button.pack(side=tk.LEFT, padx=5)

    train_model_button = tk.Button(button_frame, text="Train Model", command=lambda: run_script(os.path.join(os.getcwd(), "src", "main", "python", "train.py")))
    train_model_button.pack(side=tk.LEFT, padx=5)

    infer_button = tk.Button(button_frame, text="Infer", command=lambda: run_script(os.path.join(os.getcwd(), "src", "main", "python", "infer.py")))
    infer_button.pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()

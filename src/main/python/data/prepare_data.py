import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(labels_file):
    df = pd.read_csv(labels_file)
    df['video_path'] = df['Stealing002_x264_0'] 
    df = df.drop(columns=['Stealing002_x264_0', 'Stealing'])
    df = df.rename(columns={'0': 'label'})
    return df

def prepare_data(labels_file, output_dir, base_video_dir, log_callback=None):
    if log_callback:
        log_callback("Loading data...")
    df = load_data(labels_file)
    
    df['video_path'] = df['video_path'].apply(lambda x: os.path.join(base_video_dir, f"{x.split('_')[0]}_x264.mp4", x))
    
    if log_callback:
        log_callback("Splitting data...")
    train, temp = train_test_split(df, test_size=0.4, random_state=42, stratify=df['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])
    
    if log_callback:
        log_callback(f"Number of training samples: {len(train)}")
        log_callback(f"Number of validation samples: {len(val)}")
        log_callback(f"Number of test samples: {len(test)}")
    
    if log_callback:
        log_callback("Saving data splits...")
    save_splits(train, val, test, output_dir)
    
    if log_callback:
        log_callback(f"Data splits saved in {output_dir}")

def save_splits(train, val, test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for split, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
        split['video_path'] = split['video_path'].apply(lambda x: os.path.abspath(x))
        split.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)
        print(f"Saved {len(split)} entries to {name}.csv")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python prepare_data.py <labels_file> <output_directory> <base_video_directory>")
        sys.exit(1)

    labels_file = sys.argv[1]
    output_dir = sys.argv[2]
    base_video_dir = sys.argv[3]
    prepare_data(labels_file, output_dir, base_video_dir, print)
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(labels_file):
    df = pd.read_csv(labels_file)
    df['video_path'] = df['Shoplifting001_x264_0']
    df = df.drop(columns=['Shoplifting001_x264_0', 'Shoplifting'])
    df = df.rename(columns={'0': 'label'})
    df['video_path'] = df['video_path'].apply(lambda x: f"{x}.mp4")
    return df

def split_data(df, test_size=0.2, val_size=0.2):
    train_val, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=42, stratify=train_val['label'])
    return train, val, test

def save_splits(train, val, test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for split, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
        split['video_path'] = split['video_path'].apply(lambda x: os.path.abspath(x))
        split.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)
        print(f"Saved {len(split)} entries to {name}.csv")

def prepare_data(labels_file, output_dir, log_callback=None):
    if log_callback:
        log_callback("Loading data...")
    df = load_data(labels_file)
    
    if log_callback:
        log_callback("Splitting data...")
    train, val, test = split_data(df)
    
    if log_callback:
        log_callback(f"Number of training samples: {len(train)}")
        log_callback(f"Number of validation samples: {len(val)}")
        log_callback(f"Number of test samples: {len(test)}")
    
    if log_callback:
        log_callback("Saving data splits...")
    save_splits(train, val, test, output_dir)
    
    if log_callback:
        log_callback(f"Data splits saved in {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_data.py <labels_file> <output_directory>")
        sys.exit(1)

    labels_file = sys.argv[1]
    output_dir = sys.argv[2]
    prepare_data(labels_file, output_dir, print)
import os
import json
import argparse
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
import sys

# Add backend to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.train import ECGNet
from preprocessing.data_loader import load_data
from preprocessing.dsp import apply_butterworth_filter, normalize_signal

def evaluate_pipeline(model, X, y, device, desc=""):
    start_time = time.time()
    inputs = torch.tensor(X).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    end_time = time.time()
    
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average='macro')
    latency = (end_time - start_time) * 1000 / len(y) # ms per beat
    
    print(f"[{desc}] Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Latency: {latency:.3f} ms/beat")
    return {"accuracy": acc, "macro_f1": macro_f1, "latency_ms_per_beat": latency}

def run_ablation(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGNet(num_classes=5).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    print("Loading test data...")
    # Assume data_loader handles loading raw records and annotations if we don't use cache,
    # but since we might only have preprocessed cached data, let's load what we can
    # and simulate the ablation by applying transformations, OR if load_data gives raw data:
    _, _, (X_test, y_test) = load_data(args.data_path, seed=args.seed)
    X_test = X_test.astype(np.float32)
    
    results = {}
    
    # 1. Raw Data (simulated if X_test is already processed, but let's assume it's somewhat raw or we just evaluate as-is for baseline if we can't revert)
    # Ideally, X_test here is raw segmented beats. 
    print("\nEvaluating Condition 1: Raw Data")
    results['Raw'] = evaluate_pipeline(model, X_test, y_test, device, "Raw Data")
    
    # 2. Filtered (Butterworth)
    print("\nEvaluating Condition 2: Butterworth Filter Only")
    X_filtered = np.array([apply_butterworth_filter(beat.flatten()).reshape(1, -1) for beat in X_test], dtype=np.float32)
    results['Filtered'] = evaluate_pipeline(model, X_filtered, y_test, device, "Filtered")
    
    # 3. Full Pipeline (Butterworth + Normalization)
    print("\nEvaluating Condition 3: Full Preprocessing Pipeline")
    X_full = np.array([normalize_signal(beat.flatten()).reshape(1, -1) for beat in X_filtered], dtype=np.float32)
    results['Full Pipeline'] = evaluate_pipeline(model, X_full, y_test, device, "Full Pipeline")
    
    # Save results
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Plot ablation results
    df = pd.DataFrame(results).T
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df.index))
    width = 0.35
    
    ax1.bar(x - width/2, df['accuracy'], width, label='Accuracy', color='skyblue')
    ax1.bar(x + width/2, df['macro_f1'], width, label='Macro F1', color='salmon')
    
    ax1.set_ylabel('Score')
    ax1.set_title('Ablation Study: Preprocessing Pipeline Impact')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.index)
    ax1.set_ylim([0, 1.1])
    ax1.legend(loc='upper left')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(args.save_path.replace('.json', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAblation results saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../datasets/mit-bih')
    parser.add_argument('--model_path', type=str, default='models/ecg_cnn.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='results/ablation_study.json')
    
    run_ablation(parser.parse_args())

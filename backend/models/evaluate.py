import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from train import ECGNet

def evaluate(args):
    # Enforce reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGNet(num_classes=5).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Warning: Model not found at {args.model_path}. Using random weights for evaluation test.")
        
    model.eval()
    
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from preprocessing.data_loader import load_data
    
    print(f"Loading actual test data from {args.data_path}...")
    _, _, (X_test, y_test) = load_data(args.data_path, seed=args.seed)
    X_test = X_test.astype(np.float32)
    
    # Run Inference
    with torch.no_grad():
        inputs = torch.tensor(X_test).to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
    class_names = ['N', 'V', 'A', 'L', 'R']
    
    # Calculate Statistical Metrics
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')
    weighted_f1 = f1_score(y_test, preds, average='weighted')
    report_dict = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)
    
    metrics = {
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": report_dict
    }
    
    # Save Metrics JSON
    os.makedirs(os.path.dirname(args.metrics_save_path), exist_ok=True)
    with open(args.metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('ECG Arrhythmia Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    os.makedirs(os.path.dirname(args.cm_save_path), exist_ok=True)
    plt.savefig(args.cm_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nEvaluation Complete.")
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    print(f"-> Metrics JSON saved to {args.metrics_save_path}")
    print(f"-> Confusion Matrix saved to {args.cm_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained ECG Model")
    parser.add_argument('--data_path', type=str, default='../datasets/mit-bih')
    parser.add_argument('--model_path', type=str, default='../models/best_model.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metrics_save_path', type=str, default='../reports/results/metrics.json')
    parser.add_argument('--cm_save_path', type=str, default='../reports/results/confusion_matrix.png')
    evaluate(parser.parse_args())

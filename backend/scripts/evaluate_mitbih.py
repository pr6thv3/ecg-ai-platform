import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, roc_curve, auc
import sys

# Add backend to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.train import ECGNet
from preprocessing.data_loader import load_data

def evaluate(args):
    # Ensure reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGNet(num_classes=5).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        #print(f"Loaded model from {args.model_path}")
    else:
        #print(f"Warning: Model not found at {args.model_path}. Using random weights.")
        pass
        
    model.eval()
    
    _, _, (X_test, y_test) = load_data(args.data_path, seed=args.seed)
    X_test = X_test.astype(np.float32)
    
    # Run Inference
    with torch.no_grad():
        inputs = torch.tensor(X_test).to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
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
        
    # Generate and Save Confusion Matrix Plot (Normalized by true class)
    cm = confusion_matrix(y_test, preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('ECG Arrhythmia Confusion Matrix (Normalized by True Class)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    os.makedirs(os.path.dirname(args.cm_save_path), exist_ok=True)
    plt.savefig(args.cm_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate and Save ROC Curves
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_test == i).astype(int)
        y_score = probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {class_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (One-vs-Rest)')
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(args.roc_save_path), exist_ok=True)
    plt.savefig(args.roc_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print formatted text summary
    print(f"{'Class':<6}| {'Precision':<9}| {'Recall':<6}| {'F1':<5}| {'Support':<7}")
    print("-" * 42)
    for cls in class_names:
        cls_metrics = report_dict[cls]
        print(f"{cls:<6}| {cls_metrics['precision']:.3f}     | {cls_metrics['recall']:.3f}  | {cls_metrics['f1-score']:.3f} | {int(cls_metrics['support'])}")
    
    print(f"\nOverall accuracy: {acc * 100:.1f}%")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained ECG Model")
    parser.add_argument('--data_path', type=str, default='../datasets/mit-bih')
    parser.add_argument('--model_path', type=str, default='models/ecg_cnn.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metrics_save_path', type=str, default='results/metrics.json')
    parser.add_argument('--cm_save_path', type=str, default='results/confusion_matrix.png')
    parser.add_argument('--roc_save_path', type=str, default='results/roc_curves.png')
    
    # Make sure we run from backend root if needed, or paths are absolute.
    # The default paths assume running from backend directory.
    evaluate(parser.parse_args())

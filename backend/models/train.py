import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# 1D CNN Architecture for physiological time series
class ECGNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGNet, self).__init__()
        # Input: (batch, 1, 360)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> 180
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> 90
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def train(args):
    # Ensure fully reproducible runs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGNet(num_classes=5).to(device)
    
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from preprocessing.data_loader import load_data
    
    print(f"Loading actual data from {args.data_path}...")
    (X_train, y_train), (X_val, y_val), _ = load_data(args.data_path, seed=args.seed)
    X_train, X_val = X_train.astype(np.float32), X_val.astype(np.float32)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)), batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Cosine Annealing reduces LR gracefully
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Why TensorBoard over WandB?
    # TensorBoard runs completely locally. In biomedical/clinical applications, 
    # preventing internal model metrics from leaving the hospital's intranet via 3rd party APIs is highly preferred.
    writer = SummaryWriter(log_dir=args.log_dir)
    
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        # Validation Loop
        model.eval()
        val_preds, val_targets, val_loss = [], [], 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        # Compute Metrics
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1_macro = f1_score(val_targets, val_preds, average='macro')
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1_macro:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/Validation_Accuracy', val_acc, epoch)
        writer.add_scalar('Metrics/Validation_F1_Macro', val_f1_macro, epoch)
        
        # Checkpointing & Early Stopping
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), args.model_save_path)
            patience_counter = 0
            print("  --> Saved new best checkpoint!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in validation F1.")
                break

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D CNN for ECG Classification")
    parser.add_argument('--data_path', type=str, default='../datasets/mit-bih')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10, help='Patience for EarlyStopping')
    parser.add_argument('--log_dir', type=str, default='../logs/tensorboard')
    parser.add_argument('--model_save_path', type=str, default='../models/best_model.pth')
    train(parser.parse_args())

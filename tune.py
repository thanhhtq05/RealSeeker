"""
tune.py — BƯỚC 6 (Tối ưu hóa mô hình)

Dùng Optuna để search hyperparameter thay vì grid search thủ công.
Cài: pip install optuna

Chạy: python tune.py --data-root data --n-trials 20
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from dataset import get_dataloaders
from model import FakeDetectorCNN, FakeDetectorDualBranch
from train import train_one_epoch, validate


def objective(trial, data_root, device, n_epochs=10):
    # --- Không gian search ---
    model_name = trial.suggest_categorical("model", ["baseline", "dual_branch"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    train_loader, test_loader, _ = get_dataloaders(data_root, batch_size=batch_size)

    if model_name == "baseline":
        model = FakeDetectorCNN(dropout=dropout).to(device)
    else:
        model = FakeDetectorDualBranch(dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    for epoch in range(n_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc = validate(model, test_loader, criterion, device)
        best_val_acc = max(best_val_acc, val_acc)

        # Cho Optuna khả năng dừng sớm những trial không triển vọng (pruning)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs-per-trial", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )
    study.optimize(
        lambda trial: objective(trial, args.data_root, device, args.epochs_per_trial),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 60)
    print(f"Best val_acc: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Lưu lại toàn bộ lịch sử trial để phân tích thêm (bước 7)
    df = study.trials_dataframe()
    df.to_csv("tuning_results.csv", index=False)
    print("\nĐã lưu chi tiết các trial vào tuning_results.csv")


if __name__ == "__main__":
    main()

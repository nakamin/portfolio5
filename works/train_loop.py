import mlflow
import timm
import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils.logger import log_metrics
from models.model_factory import create_model

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    train_correct = 0
    total_samples = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        train_loss += loss.item() * batch_size 
        train_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += batch_size
    
    train_loss /= total_samples
    train_acc = train_correct/ total_samples
    
    return train_loss, train_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    total_val_samples = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            #labels = labels.squeeze().long()  # ラベルを1Dに変換
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            val_correct += (outputs.argmax(1) == labels).sum().item()
            total_val_samples += batch_size
        
    val_loss /= total_val_samples
    val_acc = val_correct / total_val_samples
        
    return val_loss, val_acc

def objective(cfg, train_loader, val_loader, num_classes, num_epochs, model_filename):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer =optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)

    best_val_loss = float("inf")
    best_epoch = 0
    best_acc = 0
    best_model_state = None

    # ダミー入力データ（モデルの入力形式を記録するだけ）を作成（バッチサイズ1, チャンネル3, 224x224画像）
    input_example = torch.randn(1, 3, 224, 224).to(torch.float32).cpu().numpy() # mlflowはnumpy配列は受け付ける

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

        # ベストモデルを更新
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_acc = val_acc
            best_model_state = model.state_dict() # ここで更新
        
        scheduler.step()

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # ベストモデルの状態をロード
    model.load_state_dict(best_model_state)
    # ベストモデルを保存
    torch.save(best_model_state, model_filename) # モデルの状態を.pthファイルに記録
    mlflow.log_artifact(model_filename) # MLflow にファイルとして記録  
    # 最終的なbest_model_stateをログする
    mlflow.pytorch.log_model(model, "best_model", input_example=input_example)

    return best_epoch, best_val_loss, best_acc, best_model_state  
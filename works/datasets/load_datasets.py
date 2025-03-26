import torch
import hydra
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import psutil

from datasets.data_transforms import train_transform, val_transform

def metadata(cfg):
    # Hydraで移動後でも、元のディレクトリを取得する
    root_dir = hydra.utils.get_original_cwd()
    metadata_path = os.path.join(root_dir, cfg.dataset.metadata_csv)
    metadata = pd.read_csv(metadata_path)
    metadata_labels = metadata[['image_id', 'dx']]
    labels_mapping = metadata_labels['dx'].map({"akiec":0, "bcc":1, "bkl":2, "df":3, "mel":4, "nv":5, "vasc":6})
    #image_id = metadata_labels["image_id"].values
    image_id = metadata_labels['image_id'].astype(str).tolist()
    labels = labels_mapping.values
    return image_id, labels

# ImageDataset クラスを定義し、データローダーで取得
class ImageDataset(Dataset):
    def __init__(self, image_id, labels, image_dir, transform=None):
        self.image_id = [f if f.endswith('.jpg') else f + ".jpg" for f in image_id] # 拡張子を追加
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_id)
    
    def __getitem__(self, idx):
        # 画像を指定するディレクトリを作成
        root_dir = hydra.utils.get_original_cwd()
        img_path = os.path.join(root_dir, self.image_dir, self.image_id[idx]) # self.image_id[idx]は画像ファイル名
        
        # ファイルが存在しない場合のエラー
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")
        
        # ディレクトリで画像を指定しPIL形式で開き、#チャンネルに変換する
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
    
# CustomSubsetクラス
class CustomSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]] # 元のデータセットから取得
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
# Datasetの作成からDataloaderの作成まで
def make_dataloader(image_id, labels, train_idx, val_idx, cfg):
    # 画像とラベルのデータセットを作成
    dataset = ImageDataset(image_id, labels, image_dir=cfg.dataset.image_dir, transform=None)

    # データセットの分割
    train_dataset = CustomSubset(dataset, train_idx, transform=train_transform)
    val_dataset = CustomSubset(dataset, val_idx, transform=val_transform)

    # Dataloaderの作成
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    """
    for imgs, labels in train_loader:
        print(f"Train Batch shape: {imgs.shape}, Labels: {labels}")
        break

    for imgs, labels in val_loader:
        print(f"Val Batch shape: {imgs.shape}, Labels: {labels}")
        break

    print(f"使用中メモリ: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
    """
    return train_loader, val_loader
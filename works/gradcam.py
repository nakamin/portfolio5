import numpy as np
import matplotlib.pyplot as plt 
import torch
import timm
import hydra
from omegaconf import DictConfig
from datasets.load_datasets import metadata, ImageDataset, CustomSubset
from sklearn.model_selection import train_test_split
from models.model_factory import create_model
from datasets.data_transforms import val_transform
from gradcam_utils import get_valdata, get_heatmap, visualize_heatmap

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # メタデータとラベル読み込み
    image_id, labels = metadata(cfg)

    # validationデータのインデックス取得（再現のために seed 使用）
    _, val_idx = train_test_split(
        range(len(image_id)),
        test_size=0.3,
        stratify=labels,
        random_state=cfg.experiment.seed
    )

    model = timm.create_model('legacy_xception', pretrained=False, num_classes=7)
    model_save_path = "/works/multirun/2025-03-25/07-37-10/3/best_model_2503_skin_cancer_classification_lr=0.0005_bs=32.pth"
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    dataset = ImageDataset(image_id, labels, image_dir=cfg.dataset.image_dir, transform=None)
    val_subset = CustomSubset(dataset, val_idx, transform=val_transform)
    correct_samples = get_valdata(val_idx, dataset, image_id, device, model)

    heatmaps, correct_samples = get_heatmap(cfg, correct_samples, val_idx, val_subset, device, model)
    visualize_heatmap(cfg, correct_samples, heatmaps)

if __name__ == '__main__':
    main()
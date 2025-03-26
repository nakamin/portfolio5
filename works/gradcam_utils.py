import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI非依存の非対話型バックエンド

import matplotlib.pyplot as plt
import cv2
import torch
import os
import hydra
from collections import defaultdict
from PIL import Image
from datasets.data_transforms import val_transform

def get_valdata(val_idx, dataset, image_id, device, model):
    correct_samples = dict() # {class_id: (image_path, true_label, pred_label)}

    # 元画像取得
    image_path = "/works/data/images"

    for idx in val_idx:
        true_label = dataset.labels[idx]
        if true_label in correct_samples:
            continue # 正解画像を記録していたらスキップ

        # 元画像を読み込み
        image_id = [f if f.endswith('.jpg') else f + ".jpg" for f in image_id]
        image_full_path = os.path.join(image_path, image_id[idx])
        pil_image = Image.open(image_full_path).convert("RGB")

        # 前処理→テンソルにして推論
        image_tensor = val_transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            pred_label = torch.argmax(output, dim=1).item()

        # 正解していたら辞書に保存
        if pred_label == true_label or true_label == 3:
            # キーはクラスID（true_label）、値は (画像, 正解ラベル, 予測ラベル)
            correct_samples[true_label] = (pil_image, true_label, pred_label)
            print(f"[idx {idx}] True: {true_label}, Pred: {pred_label}")
        
        if len(correct_samples) == 7:
            break
    return correct_samples

def get_target_layer(model, layer_path: str):
    parts = layer_path.split(".")
    layer = model  # モデル本体からスタート
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    return layer  # ← 最終的な層（Module）を返す

# 特徴量と勾配を取得し、ヒートマップを作成
class GradCam:
    def __init__(self, model, target):
        self.model = model.eval() # 推論モード
        self.feature = None # forward hookで取得する特徴マップの格納先
        self.gradient = None # backward hookで取得する勾配マップの格納先
        self.target = target # Grad-CAMをかけたい層
        self._get_hook() # hook関数をtargetに登録

    # 順伝番から特徴量を得る
    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output) # [batch, tokens, dim] → [batch, channels, H, W] に変換
    # 逆伝播から特徴量を得る
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad[0]) # 勾配テンソル（＝backwardで流れてきた値）を変換

        # def _store_grad(grad):
            #self.gradient = self.reshape_transform(grad)
        
        #output_grad[0].register_hook(_store_grad) # 勾配を保存するフックをさらに登録
    
    # targetにforward/backwardの両方のhookを登録
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_full_backward_hook(self._get_grads_hook)

    # 可視化のための成型
    def reshape_transform(self, tensor, height=None, width=None):
        #入力を変換
        #tensor = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2)) # CLSトークンを除外 → [B, 196, D(768)] → [B, H, W, C]で空間構造に並べなおす
        #tensor = tensor.permute(0, 3, 1, 2) # CNNと同じ形式[B, C, H, W]に並べ替え
        return tensor

    # Grad-CAMの計算(推論 → 逆伝播 → CAM作成（勾配×特徴量 → 加重和 → ReLU → 正規化）)
    def __call__(self, inputs):
        device = next(self.model.parameters()).device # モデルと同じデバイスを取得
        inputs = inputs.to(device)

        self.model.zero_grad()
        output = self.model(inputs) # 入力画像でforward実行
        
        # 最もスコアが高いクラスのindexを取得
        index = np.argmax(output.cpu().data.numpy())
        
        # 最もスコアが高いクラスの出力を backward
        target = output[0][index]
        target.backward()

        # 勾配と特徴マップの取得
        gradient = self.gradient[0].detach().numpy()
        feature = self.feature[0].detach().numpy()

        # 各チャネルの勾配を平均化（重要度スコア）
        weight = np.mean(gradient, axis=(1,2))
        
        # チャネルごとに重み付き和を計算
        cam = np.sum(feature * weight[:, np.newaxis, np.newaxis], axis=0)
        cam = np.maximum(cam, 0) # ReLU

        # 正規化
        cam -= np.min(cam)
        cam /= np.max(cam + 1e-8)
        # リサイズ
        cam = cv2.resize(cam, (224, 224))
        return cam
    
def get_heatmap(cfg, correct_samples, val_idx, val_subset, device, model):
    heatmaps = {}
    # 各クラスの画像に対してGrad-CAMを計算
    for class_id, (pil_image, true_label, pred_label) in correct_samples.items():   
        image_tensor = val_transform(pil_image).unsqueeze(0)

        target_layer = get_target_layer(model, cfg.train.target_layer)
        # モデルに入力してGrad-CAMマスクを生成
        grad_cam = GradCam(model, target_layer) # インスタンス生成
        cam_mask = grad_cam(image_tensor) # __call__() が発動

        # 正規化
        #heatmap = (cam_mask - cam_mask.min()) / (cam_mask.max() - cam_mask.min()) 0で割り算が発生
        cam_min = cam_mask.min()
        cam_max = cam_mask.max()
        denom = cam_max - cam_min

        if denom == 0:
            heatmap = np.zeros_like(cam_mask) # 全部ゼロのマップ
        else:
            heatmap = (cam_mask - cam_min) / denom

        heatmap = np.nan_to_num(heatmap) # NaN 除去
        heatmaps[class_id] = cam_mask

    return heatmaps, correct_samples

def visualize_heatmap(cfg, correct_samples, heatmaps):
        
    for class_id, (pil_img, true_label, pred_label) in correct_samples.items():  
        heatmap = heatmaps[class_id]

        image_np = np.array(pil_img.resize((224, 224))) # PIL → Numpy
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # RGB → BGR

        # camをヒートマップカラーに変換
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # ヒートマップをリサイズして元画像に合わせる
        heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))

        # ヒートマップと画像を合成
        superimposed = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
        
        plt.figure(figsize=(6,6))
        plt.title(f"Class {class_id} | True: {true_label} | Pred: {pred_label}")
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)) # RGBに戻す
        plt.axis('off')

        output_dir = hydra.utils.to_absolute_path(cfg.gradcam.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"gradcam_class_{class_id}.png"))
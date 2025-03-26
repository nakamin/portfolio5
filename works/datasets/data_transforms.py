from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((int(299/ 0.875)), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(299),
    transforms.RandomRotation(10),
    transforms.ToTensor(), # PIL 画像を (C, H, W) の torch.Tensor に変換
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
val_transform = transforms.Compose([
    transforms.Resize((int(299/ 0.875)), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
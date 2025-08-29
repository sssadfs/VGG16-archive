import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BATCH_SIZE = 32

transformation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(label_file, 'r') as f:    # 'r'是读文本模式
            lines = f.readlines()
            for line in lines:
                path, label = line.strip().split(' ')   # 按' '分割
                full_path = os.path.join(root_dir, path.lstrip("./")) # 拼接路径去掉txt文件中的./
                self.samples.append((full_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = CustomDataset(root_dir="../tiny-imagenet-200/",label_file="../tiny-imagenet-200/train.txt", transform=transformation)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = CustomDataset(root_dir="../tiny-imagenet-200/",label_file="../tiny-imagenet-200/val.txt",transform=transformation)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

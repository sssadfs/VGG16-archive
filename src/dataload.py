import torch
import torchvision
from torchvision import transforms

BATCH_SIZE = 32

transformation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.ImageFolder(root = "../tiny-imagenet-200/train",transform=transformation)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = torchvision.datasets.ImageFolder(root = "../tiny-imagenet-200/val",transform=transformation)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

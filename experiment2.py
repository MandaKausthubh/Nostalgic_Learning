import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.models import ImageClassifier
from tqdm import tqdm

from time import time

import torch
from typing import Optional, Tuple




# preparing transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# CONFIGURATION:
ROOT_DIR = '/Users/mandakausthubh/data'


# DATASETS:

# CIFAR10 datasets
cifar10_train_dataset = datasets.CIFAR10(root=f'{ROOT_DIR}', train=True, download=True, transform=transform)
cifar10_val_dataset = datasets.CIFAR10(root=f'{ROOT_DIR}', train=False, download=True, transform=transform)
cifar10_train_loader = DataLoader(cifar10_train_dataset, batch_size=64, shuffle=True)
cifar10_val_loader = DataLoader(cifar10_val_dataset, batch_size=64, shuffle=False)

# CIFAR-100 datasets
cifar100_train_dataset = datasets.CIFAR100(root=f'{ROOT_DIR}', train=True, download=True, transform=transform)
cifar100_val_dataset = datasets.CIFAR100(root=f'{ROOT_DIR}', train=False, download=True, transform=transform)
cifar100_train_loader = DataLoader(cifar100_train_dataset, batch_size=64, shuffle=True)
cifar100_val_loader = DataLoader(cifar100_val_dataset, batch_size=64, shuffle=False)

# # STL-10 datasets
# stl10_train_dataset = datasets.STL10(root=f'{ROOT_DIR}', split='train', download=True, transform=transform)
# stl10_val_dataset = datasets.STL10(root=f'{ROOT_DIR}', split='test', download=True, transform=transform)
# stl10_train_loader = DataLoader(stl10_train_dataset, batch_size=64, shuffle=True)
# stl10_val_loader = DataLoader(stl10_val_dataset, batch_size=64, shuffle=False)

# # Caltech-256 datasets
# caltech256_train_dataset = datasets.Caltech256(root=f'{ROOT_DIR}/caltech256', download=True, transform=transform)
# caltech256_val_dataset = datasets.Caltech256(root=f'{ROOT_DIR}/caltech256', download=True, transform=transform)
# caltech256_train_loader = DataLoader(caltech256_train_dataset, batch_size=64, shuffle=True)
# caltech256_val_loader = DataLoader(caltech256_val_dataset, batch_size=64, shuffle=False)

# # Tiny ImageNet datasets
# tiny_imagenet_train_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/tiny-imagenet-200/train', transform=transform)
# tiny_imagenet_val_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/tiny-imagenet-200/val', transform=transform)
# tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=64, shuffle=True)
# tiny_imagenet_val_loader = DataLoader(tiny_imagenet_val_dataset, batch_size=64, shuffle=False)


imageClassifier = ImageClassifier().to('mps')

imageClassifier.add_task('CIFAR10', num_classes=10)
imageClassifier.set_active_task('CIFAR10')

imageClassifier.add_task('CIFAR100', num_classes=100)
imageClassifier.set_active_task('CIFAR100')


# Number of trainable backbone parameters
num_params = sum(
    p.numel()
    for _, p in imageClassifier.backbone.named_parameters()
    if p.requires_grad
)

k = 32  # or any small value for testing

Q_test = torch.rand(
    num_params,
    k,
    device=imageClassifier.device,
    dtype=next(
        p.dtype
        for _, p in imageClassifier.backbone.named_parameters()
        if p.requires_grad
    ),
)
# Q_test, _ = torch.linalg.qr(Q_test, mode="reduced")

imageClassifier.set_Q(Q_test, scaling=None)
imageClassifier.set_active_task('CIFAR10')

optimizer = imageClassifier.configure_optimizers()



# Testing a single training step with projection
for input, target in (cifar10_train_loader):
    input, target = input.to('mps'), target.to('mps')

    start_time = time()
    optimizer.zero_grad()
    output = imageClassifier(input)
    loss = imageClassifier.criterion(output, target)
    loss.backward()
    optimizer.step()
    end_time = time()
    print(f"Step time with projection: {end_time - start_time:.4f} seconds")
    break









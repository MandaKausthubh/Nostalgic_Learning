import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

import time

# import models.models as models
from models.models import ImageClassifier

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





imageClassifier = ImageClassifier(lora_r=1).to('mps')

imageClassifier.add_task('CIFAR10', num_classes=10)
imageClassifier.set_active_task('CIFAR10')

imageClassifier.add_task('CIFAR100', num_classes=100)
imageClassifier.set_active_task('CIFAR100')

print("Using PEFT: ", imageClassifier.use_peft)
print("Is PEFT on: ", imageClassifier.is_peft_on)
print("Trainable parameters: ", sum(p.numel() for p in imageClassifier.parameters() if p.requires_grad))


from utils_new.hvp import HessianVectorProduct
from utils_new.lanczos import Lanczos


imageClassifier.set_active_task('CIFAR10')

imageClassifier_trainable_params = [p for _, p in imageClassifier.backbone.named_parameters() if p.requires_grad]
hvp = HessianVectorProduct(
    model=imageClassifier,
    loss_fn=imageClassifier.criterion,
    device=torch.device('mps'),
    trainable_params=imageClassifier_trainable_params
)


# Simulate training on CIFAR-10
v = torch.randn(hvp.num_params).to('mps')

for inputs, targets in cifar10_train_loader:
    inputs.to('mps')
    targets.to('mps')

    start_time = time.time()
    hvp_result = hvp.hvp(inputs, targets, v) #type: ignore
    end_time = time.time()
    print(f"HVP computation time for one batch: {end_time - start_time} seconds")
    break


# Testing Lanczos
lanczos = Lanczos(hvp_computer=hvp, device=torch.device('mps'))

for inputs, targets in cifar10_train_loader:
    inputs.to('mps')
    targets.to('mps')

    start_time = time.time()
    T, Q = lanczos.run(inputs, targets, k=32)  #type: ignore
    end_time = time.time()

    QtQ = Q.T @ Q
    print(torch.max(torch.abs(QtQ - torch.eye(QtQ.size(0), device=QtQ.device))))


    print(f"Lanczos computation time: {end_time - start_time} seconds")
    break




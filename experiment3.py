from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.models import ImageClassifier
from tqdm import tqdm

from utils_new.accumulate import accumulate_hessian_eigenspace
from utils_new.hessian import compute_Q_for_task


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
imageClassifier.add_task('CIFAR100', num_classes=100)

optimizer = imageClassifier.configure_optimizers()
imageClassifier.set_active_task('CIFAR10')

# Testing a single training step with projection
for input, target in tqdm(cifar10_train_loader):
    input, target = input.to('mps'), target.to('mps')
    optimizer.zero_grad()
    output = imageClassifier(input)
    loss = imageClassifier.criterion(output, target)
    loss.backward()
    optimizer.step()


total_loss = 0.0
for input, target in tqdm(cifar10_val_loader):
    input, target = input.to('mps'), target.to('mps')
    output = imageClassifier(input)
    loss = imageClassifier.criterion(output, target)

    total_loss += loss.item() * input.size(0)

avg_loss = total_loss / len(cifar10_val_loader.dataset) #type: ignore
print(f'CIFAR10 Validation Loss: {avg_loss}')


Q = None
Lambda = None

# Calculating new Q and Lambda for CIFAR10
Q, Lambda = compute_Q_for_task(imageClassifier, cifar10_train_loader, device='mps', k=16)
imageClassifier.set_active_task('CIFAR100')
imageClassifier.set_Q(Q, None)

optimizer = imageClassifier.configure_optimizers()

for input, target in tqdm(cifar100_train_loader):
    input, target = input.to('mps'), target.to('mps')
    optimizer.zero_grad()
    output = imageClassifier(input)
    loss = imageClassifier.criterion(output, target)
    loss.backward()
    optimizer.step()

imageClassifier.set_active_task('CIFAR10')
total_loss = 0.0
for input, target in tqdm(cifar10_val_loader):
    input, target = input.to('mps'), target.to('mps')
    output = imageClassifier(input)
    loss = imageClassifier.criterion(output, target)

    total_loss += loss.item() * input.size(0)

avg_loss = total_loss / len(cifar10_val_loader.dataset) #type: ignore
print(f'CIFAR10 Validation Loss: {avg_loss}')




print("\n\n--- Now testing without PEFT ---\n\n")

imageClassifierWithoutPeft = ImageClassifier().to('mps')
imageClassifierWithoutPeft.add_task('CIFAR10', num_classes=10)
imageClassifierWithoutPeft.add_task('CIFAR100', num_classes=100)

optimizerWithoutPeft = imageClassifierWithoutPeft.configure_optimizers()
imageClassifierWithoutPeft.set_active_task('CIFAR10')

# Testing a single training step without projection
for input, target in tqdm(cifar10_train_loader):
    input, target = input.to('mps'), target.to('mps')
    optimizerWithoutPeft.zero_grad()
    output = imageClassifierWithoutPeft(input)
    loss = imageClassifierWithoutPeft.criterion(output, target)
    loss.backward()
    optimizerWithoutPeft.step()

#Validation for CIFAR10 without PEFT
total_loss = 0.0
for input, target in tqdm(cifar10_val_loader):
    input, target = input.to('mps'), target.to('mps')
    output = imageClassifierWithoutPeft(input)
    loss = imageClassifierWithoutPeft.criterion(output, target)

    total_loss += loss.item() * input.size(0)

avg_loss = total_loss / len(cifar10_val_loader.dataset) #type: ignore
print(f'[Without PEFT] CIFAR10 Validation Loss: {avg_loss}')

# Training on CIFAR100 without PEFT
imageClassifierWithoutPeft.set_active_task('CIFAR100')
optimizerWithoutPeft = imageClassifierWithoutPeft.configure_optimizers()
for input, target in tqdm(cifar100_train_loader):
    input, target = input.to('mps'), target.to('mps')
    optimizerWithoutPeft.zero_grad()
    output = imageClassifierWithoutPeft(input)
    loss = imageClassifierWithoutPeft.criterion(output, target)
    loss.backward()
    optimizerWithoutPeft.step()

# Validation for CIFAR10 without PEFT after training on CIFAR100
imageClassifierWithoutPeft.set_active_task('CIFAR10')
total_loss = 0.0
for input, target in tqdm(cifar10_val_loader):
    input, target = input.to('mps'), target.to('mps')
    output = imageClassifierWithoutPeft(input)
    loss = imageClassifierWithoutPeft.criterion(output, target)

    total_loss += loss.item() * input.size(0)

avg_loss = total_loss / len(cifar10_val_loader.dataset) #type: ignore
print(f'[Without PEFT] CIFAR10 Validation Loss: {avg_loss}')







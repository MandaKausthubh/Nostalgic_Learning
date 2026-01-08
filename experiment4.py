import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.models import ImageClassifier
from tqdm import tqdm


# preparing transforms
transform = transforms.Compose([
    transforms.Resize(72),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),

    transforms.RandAugment(
        num_ops=2,
        magnitude=9
    ),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.25),
        ratio=(0.3, 3.3),
        value='random'  #type: ignore
    ),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# CONFIGURATION:
ROOT_DIR = '/Users/mandakausthubh/data'


# DATASETS:
print("Downloading and preparing datasets...")

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

# STL-10 datasets
stl10_train_dataset = datasets.STL10(root=f'{ROOT_DIR}', split='train', download=True, transform=transform)
stl10_val_dataset = datasets.STL10(root=f'{ROOT_DIR}', split='test', download=True, transform=transform)
stl10_train_loader = DataLoader(stl10_train_dataset, batch_size=64, shuffle=True)
stl10_val_loader = DataLoader(stl10_val_dataset, batch_size=64, shuffle=False)

# Caltech-256 datasets
caltech256_train_dataset = datasets.Caltech256(root=f'{ROOT_DIR}/caltech256', download=True, transform=transform)
caltech256_val_dataset = datasets.Caltech256(root=f'{ROOT_DIR}/caltech256', download=True, transform=transform)
caltech256_train_loader = DataLoader(caltech256_train_dataset, batch_size=64, shuffle=True)
caltech256_val_loader = DataLoader(caltech256_val_dataset, batch_size=64, shuffle=False)

# Tiny ImageNet datasets
tiny_imagenet_train_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_val_dataset = datasets.ImageFolder(root=f'{ROOT_DIR}/tiny-imagenet-200/val', transform=transform)
tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=64, shuffle=True)
tiny_imagenet_val_loader = DataLoader(tiny_imagenet_val_dataset, batch_size=64, shuffle=False)

print("Datasets/loaders ready.")



device = 'mps'
imageClassifier = ImageClassifier().to(device)

tasks = ['CIFAR10','CIFAR100','STL10', 'Caltech256', 'TinyImageNet']
task_loaders = {
    'CIFAR10': (cifar10_train_loader, cifar10_val_loader),
    'CIFAR100': (cifar100_train_loader, cifar100_val_loader),
    'STL10': (stl10_train_loader, stl10_val_loader),
    'Caltech256': (caltech256_train_loader, caltech256_val_loader),
    'TinyImageNet': (tiny_imagenet_train_loader, tiny_imagenet_val_loader),
}

task_num_classes = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'STL10': 10,
    'Caltech256': 256,
    'TinyImageNet': 200,
}

task_epochs = {
    'CIFAR10': 5,
    'CIFAR100': 5,
    'STL10': 5,
    'Caltech256': 5,
    'TinyImageNet': 5,
}


# Adding tasks to the model
for task in tasks:
    num_classes = task_num_classes[task]
    print(f"Adding task {task} with {num_classes} classes.")
    imageClassifier.add_task(task, num_classes=num_classes)

# Writer for TensorBoard
writer = SummaryWriter()
niter = 0

validation_interval = 10

imageClassifier.set_active_task('CIFAR10')

for epoch in range(1, 31):
    print(f"\n=== Epoch {epoch} ===\n")
    for image, target in tqdm(cifar10_train_loader, ncols=50):
        image = image.to(device)
        target = target.to(device)

        imageClassifier.set_active_task('CIFAR10')
        optimizer = imageClassifier.configure_optimizers()
        optimizer.zero_grad()
        outputs = imageClassifier(image)
        loss = imageClassifier.criterion(outputs, target)
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train_CIFAR10', loss.item(), niter)

        if niter % validation_interval == 0:
            validation_loss = 0.0
            for val_image, val_target in cifar10_val_loader:
                val_image = val_image.to(device)
                val_target = val_target.to(device)

                imageClassifier.set_active_task('CIFAR10')
                val_outputs = imageClassifier(val_image)
                val_loss = imageClassifier.criterion(val_outputs, val_target)
                validation_loss += val_loss.item() * val_image.size(0)

            val_loss = validation_loss / len(cifar10_val_loader.dataset)  # type: ignore
            writer.add_scalar('Loss/val_CIFAR10', val_loss, niter)

        niter += 1






import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.models import ImageClassifier
from tqdm import tqdm

from utils_new.accumulate import accumulate_hessian_eigenspace
from utils_new.hessian import compute_Q_for_task


# preparing transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# CONFIGURATION:
ROOT_DIR = '/Users/mandakausthubh/data'



@torch.no_grad()
def evaluate(model, loader, task_name, device):
    model.eval()
    model.set_active_task(task_name)

    total_loss = 0.0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = model.criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / total



def train_task(model, loader, optimizer, task_name, device, epochs=1):
    model.train()
    model.set_active_task(task_name)

    for _ in range(epochs):
        print(f"[Training] Task: {task_name} | Epoch: {_ + 1}/{epochs}")
        for x, y in tqdm(loader, ncols=50):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = model.criterion(model(x), y)
            loss.backward()
            optimizer.step()



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
    'CIFAR10': 10,
    'CIFAR100': 10,
    'STL10': 10,
    'Caltech256': 10,
    'TinyImageNet': 10,
}

eval_step=5
Q, Lambda = None, None

for t, task_name in enumerate(tasks, start=1):
    print(f"\n=== Training task {t}: {task_name} ===")

    imageClassifier.add_task(task_name=task_name, num_classes=task_num_classes[task_name])
    optimizer = imageClassifier.configure_optimizers()
    train_loader, val_loader = task_loaders[task_name]

    imageClassifier.set_active_task(task_name)
    imageClassifier.set_Q(Q, None)

    # Train the task
    train_task(
        model=imageClassifier,
        loader=train_loader,
        optimizer=optimizer,
        task_name=task_name,
        device=device,
        epochs=task_epochs[task_name],
    )

    # Evaluate after training
    val_loss = evaluate(
        model=imageClassifier,
        loader=val_loader,
        task_name=task_name,
        device=device,
    )

    # Evaluate previous tasks:
    for prev_task in tasks[:t-1]:
        prev_val_loader = task_loaders[prev_task][1]
        prev_val_loss = evaluate(
            model=imageClassifier,
            loader=prev_val_loader,
            task_name=prev_task,
            device=device,
        )
        print(f"[Evaluation] Previous Task: {prev_task} | Validation Loss: {prev_val_loss:.4f}")

    # Compute Q_t, Lambda_t for the current task
    Q_new, Lambda_new = compute_Q_for_task(
        model=imageClassifier,
        train_loader=train_loader,
        device=device,
        k=32,
    )

    # Accumulate into running average
    Q, Lambda = accumulate_hessian_eigenspace(
        Q_old=Q,
        Lambda_old=Lambda,
        Q_new=Q_new,
        Lambda_new=Lambda_new,
        t=t,
        k=32
    )

    print(f"[Evaluation] Task: {task_name} | Validation Loss: {val_loss:.4f}")



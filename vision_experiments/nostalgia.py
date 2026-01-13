import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional

from utils_new.accumulate import accumulate_hessian_eigenspace
from utils_new.hessian import compute_Q_for_task

from models.vit32 import ImageClassifierViT
from tqdm import tqdm
import datetime

from torch.utils.tensorboard import SummaryWriter




@dataclass
class NostalgiaConfig:
    seed: int = 42
    root_dir: str = '/Users/mandakausthubh/data'
    batch_size: int = 64
    learning_rate: float = 1e-4
    device: str = 'mps'
    log_dir: str = f'./logs/nostalgia_vision_experiment/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    validate_after_steps: int = 300
    hessian_eigenspace_dim: int = 16



class NostalgiaExperiment:
    def __init__(self, config: NostalgiaConfig):
        self.config = config

        torch.manual_seed(self.config.seed)

        self.transform = transforms.Compose([
            transforms.Resize(72),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self.imageClassifier = ImageClassifierViT().to(self.config.device)
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        self.finished_datasets = []



    def load_model(self, path: Optional[str] = None):
        if path is not None:
            self.imageClassifier.load_state_dict(
                torch.load(path, map_location=self.config.device)
            )
        else:
            print("No pre-trained model path provided. Using random initialization.")
            ImageClassifierViT().to(self.config.device)


    def save_model(self, path: str):
        torch.save(self.imageClassifier.state_dict(), path)



    def prepare_dataloader(self, dataset_class, train: bool = True):
        dataset = dataset_class(
            root=self.config.root_dir,
            train=train,
            download=True,
            transform=self.transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=train
        )
        return dataloader



    def prepare_all_datasets(self):
        #TODO: Implement dataset preparation logic
        # CIFAR10 datasets
        cifar10_train_dataset = datasets.CIFAR10(root=f'{self.config.root_dir}', train=True, download=True, transform=self.transform)
        cifar10_val_dataset = datasets.CIFAR10(root=f'{self.config.root_dir}', train=False, download=True, transform=self.transform)
        cifar10_train_loader = DataLoader(cifar10_train_dataset, batch_size=64, shuffle=True)
        cifar10_val_loader = DataLoader(cifar10_val_dataset, batch_size=64, shuffle=False)

        # CIFAR-100 datasets
        cifar100_train_dataset = datasets.CIFAR100(root=f'{self.config.root_dir}', train=True, download=True, transform=self.transform)
        cifar100_val_dataset = datasets.CIFAR100(root=f'{self.config.root_dir}', train=False, download=True, transform=self.transform)
        cifar100_train_loader = DataLoader(cifar100_train_dataset, batch_size=64, shuffle=True)
        cifar100_val_loader = DataLoader(cifar100_val_dataset, batch_size=64, shuffle=False)

        # STL-10 datasets
        stl10_train_dataset = datasets.STL10(root=f'{self.config.root_dir}', split='train', download=True, transform=self.transform)
        stl10_val_dataset = datasets.STL10(root=f'{self.config.root_dir}', split='test', download=True, transform=self.transform)
        stl10_train_loader = DataLoader(stl10_train_dataset, batch_size=64, shuffle=True)
        stl10_val_loader = DataLoader(stl10_val_dataset, batch_size=64, shuffle=False)

        # Caltech-256 datasets
        caltech256_train_dataset = datasets.Caltech256(root=f'{self.config.root_dir}/caltech256', download=True, transform=self.transform)
        caltech256_val_dataset = datasets.Caltech256(root=f'{self.config.root_dir}/caltech256', download=True, transform=self.transform)
        caltech256_train_loader = DataLoader(caltech256_train_dataset, batch_size=64, shuffle=True)
        caltech256_val_loader = DataLoader(caltech256_val_dataset, batch_size=64, shuffle=False)

        # Tiny ImageNet datasets
        tiny_imagenet_train_dataset = datasets.ImageFolder(root=f'{self.config.root_dir}/tiny-imagenet-200/train', transform=self.transform)
        tiny_imagenet_val_dataset = datasets.ImageFolder(root=f'{self.config.root_dir}/tiny-imagenet-200/val', transform=self.transform)
        tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=64, shuffle=True)
        tiny_imagenet_val_loader = DataLoader(tiny_imagenet_val_dataset, batch_size=64, shuffle=False)

        self.datasets = {
            'CIFAR10': (cifar10_train_loader, cifar10_val_loader),
            'CIFAR100': (cifar100_train_loader, cifar100_val_loader),
            'STL10': (stl10_train_loader, stl10_val_loader),
            'Caltech256': (caltech256_train_loader, caltech256_val_loader),
            'TinyImageNet': (tiny_imagenet_train_loader, tiny_imagenet_val_loader),
        }

        self.dataset_num_classes = {
            'CIFAR10': 10,
            'CIFAR100': 100,
            'STL10': 10,
            'Caltech256': 256,
            'TinyImageNet': 200,
        }

        for task_name in self.datasets.keys():
            self.imageClassifier.add_task(task_name, self.dataset_num_classes[task_name])


        self.epochs_per_task = {
            'CIFAR10': 20,
            'CIFAR100': 10,
            'STL10': 10,
            'Caltech256': 10,
            'TinyImageNet': 10,
        }



    def validate_dataset(
            self,
            val_loader: DataLoader,
            criterion,
            iteration: int = 0,
            task_name: Optional[str] = None
    ):
        self.imageClassifier.set_active_task(task_name)
        total_loss = 0.0
        accuracy = 0.0
        for input, target in val_loader:
            self.imageClassifier.eval()
            input, target = input.to(self.config.device), target.to(self.config.device)
            input = self.imageClassifier.preprocess_inputs(input)
            with torch.no_grad():
                output = self.imageClassifier(input)
                loss = criterion(output, target)
                total_loss += loss.item() * input.size(0)
                accuracy += (output.argmax(dim=1) == target).sum().item()

        loss = total_loss / len(val_loader.dataset) #type: ignore
        accuracy = accuracy / len(val_loader.dataset) #type: ignore

        self.writer.add_scalar( f'validation_loss for {task_name}', loss, iteration)
        self.writer.add_scalar( f'validation_accuracy for {task_name}', accuracy, iteration)



    def train_dataset(
            self,
            task_name:str,
            Q_prev = None,
            starting_step:int=0,
            epochs: int = 1,
            log_deltas: bool = False,
        ):

        train_loader, _ = self.datasets[task_name]  #type: ignore

        self.imageClassifier.set_active_task(task_name)

        self.imageClassifier.set_Q(Q_prev)
        criterion = self.imageClassifier.criterion
        niter = starting_step


        if log_deltas:
            optimizer = self.imageClassifier.configure_optimizers(
                writter=self.writer,
                iteration=starting_step
            )
        else:
            optimizer = self.imageClassifier.configure_optimizers()

        for _ in range(epochs):
            for input, target in tqdm((train_loader), desc=f"Training on {task_name}", ncols=80):
                self.imageClassifier.train()
                input, target = input.to(self.config.device), target.to(self.config.device)
                input = self.imageClassifier.preprocess_inputs(input)

                optimizer.zero_grad()
                output = self.imageClassifier(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                accuracy = (output.argmax(dim=1) == target).float().mean().item()

                # Logging:
                self.writer.add_scalar(
                    f'{task_name}/train_loss for {task_name}',
                    loss.item(),
                    niter
                )

                self.writer.add_scalar(
                    f'{task_name}/train_accuracy for {task_name}',
                    accuracy,
                    niter 
                )

                if (niter+1) % self.config.validate_after_steps == 0:
                    self.validate(niter + starting_step, task_name)

                niter += 1



    def validate(self, iteration: int = 0, task_name: Optional[str] = None):
        """
        Conduct validation on all datasets and log results.
        """
        for task_name in self.finished_datasets:  #type: ignore
            _, val_loader = self.datasets[task_name]  #type: ignore
            self.imageClassifier.set_active_task(task_name)
            criterion = self.imageClassifier.criterion
            self.validate_dataset(val_loader, criterion, iteration, task_name)


    def train(self, erase_past=False, log_deltas=True):
        self.prepare_all_datasets()

        Q_prev, Lambda_prev = None, None

        total_steps = 0
        task_counter = 1

        if erase_past:
            self.finished_datasets = []

        for task_name, (train_loader, _) in self.datasets.items():  #type: ignore
            self.finished_datasets.append(task_name)

            epochs = self.epochs_per_task[task_name]  #type: ignore

            print(f"Starting training on {task_name} for {epochs} epochs.")

            self.train_dataset(
                task_name=task_name,
                Q_prev=Q_prev,
                starting_step=total_steps,
                epochs=epochs,
                log_deltas=log_deltas
            )

            # After training on the current task, compute the Hessian eigenspace
            Q_curr, Lambda_curr = compute_Q_for_task(
                model=self.imageClassifier,
                train_loader=train_loader,
                device=self.config.device,
                k=self.config.hessian_eigenspace_dim
            )

            Q_prev, Lambda_prev = accumulate_hessian_eigenspace(
                Q_prev, Lambda_prev,
                Q_curr, Lambda_curr,
                t=task_counter, k=self.config.hessian_eigenspace_dim
            )

            total_steps += epochs * len(train_loader)

            # Save model after each task
            self.save_model(f'nostalgia_model_after_{task_name}.pth')

            print(f"Completed training on {task_name}.")
            task_counter += 1

        print("Training completed for all tasks.")





if __name__ == "__main__": 
    config = NostalgiaConfig(
        root_dir='/Users/mandakausthubh/data',
        batch_size=64,
        learning_rate=1e-4,
        hessian_eigenspace_dim=16,
        device='mps',
        validate_after_steps=300
    )

    experiment = NostalgiaExperiment(config)
    experiment.load_model()  # Load pre-trained model if available
    experiment.train()

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict

from utils_new.accumulate import accumulate_hessian_eigenspace
from utils_new.hessian import compute_Q_for_task

from models.vit32 import ImageClassifierViT
from tqdm import tqdm
import datetime

from torch.utils.tensorboard import SummaryWriter
import argparse



def get_args():
    parser = argparse.ArgumentParser(description="Nostalgia Vision Experiment")
    parser.add_argument('--mode', type=str, default='nostalgia', help='Training mode: nostalgia, l2sp, EWC, Adam',
                        choices=['nostalgia', 'l2sp', 'EWC', 'Adam'])
    parser.add_argument('--root_dir', type=str, default='/Users/mandakausthubh/data', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for optimizer')
    parser.add_argument('--device', type=str, default='mps', help='Device to use for training (e.g., cpu, cuda, mps)',
                        choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--hessian_eigenspace_dim', type=int, default=8, help='Dimension of Hessian eigenspace')
    parser.add_argument('--validate_after_steps', type=int, default=100, help='Validation frequency in steps')
    parser.add_argument('--log_dir', type=str,
                        default=f'./logs/nostalgia_vision_experiment/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--nostalgia_dimension', type=int, default=16, help='Dimension of Hessian low-rank for nostalgia method')
    return parser.parse_args()





@dataclass
class NostalgiaConfig:
    mode: str = 'nostalgia'
    seed: int = 42
    root_dir: str = '/Users/mandakausthubh/data'
    batch_size: int = 64
    learning_rate: float = 1e-4
    device: str = 'mps'
    log_dir: str = f'./logs/nostalgia_vision_experiment/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    validate_after_steps: int = 10

    hessian_eigenspace_dim: int = 16

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_modules: Optional[list] = None



class NostalgiaExperiment:
    def __init__(self, config: NostalgiaConfig):
        self.config = config

        torch.manual_seed(self.config.seed)

        # self.transform = transforms.Compose([
        #     transforms.Resize(72),
        #     transforms.Grayscale(num_output_channels=3),
        #     transforms.RandomCrop(64, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandAugment(num_ops=2, magnitude=9),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5]*3, [0.5]*3),
        # ])

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])


        using_nostalgia = (self.config.mode == "nostalgia")

        self.imageClassifier = ImageClassifierViT(
            learning_rate=self.config.learning_rate,
            lora_r=self.config.lora_r, 
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_modules=self.config.lora_modules,
            use_nostalgia=using_nostalgia,
        ).to(self.config.device)
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        self.finished_datasets = []

        self.ewc_fisher: Dict[str, Dict[str, torch.Tensor]] = {}
        self.ewc_params: Dict[str, Dict[str, torch.Tensor]] = {}


    def store_ewc_information(
        self,
        task_name: str,
        fisher_information: Dict[str, torch.Tensor],
    ):
        self.ewc_fisher[task_name] = {
            k: v.detach().clone() for k, v in fisher_information.items()
        }

        self.ewc_params[task_name] = {
            name: p.detach().clone() for name, p in self.imageClassifier.backbone.named_parameters()
            if p.requires_grad
        }




    def load_model(self, path: Optional[str] = None):
        if path is not None:
            self.imageClassifier.load_state_dict(
                torch.load(path, map_location=self.config.device)
            )
        else:
            print("No pre-trained model path provided. Using random initialization.")
            ImageClassifierViT().to(self.config.device)
            torch.save(
                self.imageClassifier.state_dict(),
                './model_weights/nostalgia_model_pretrained.pth'
            )

        self.theta_0 = {
            name: p.detach().clone() for name, p in self.imageClassifier.backbone.named_parameters()
            if p.requires_grad
        }


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



    def EWC_regularization(
        self,
        lambda_ewc: float = 1e-4,
    ):
        if not self.ewc_fisher:
            return torch.tensor(0.0, device=self.config.device)

        loss = torch.tensor(0.0, device=self.config.device)

        for task_name in self.ewc_fisher.keys():
            fisher = self.ewc_fisher[task_name]
            theta_star = self.ewc_params[task_name]

            for name, param in self.imageClassifier.backbone.named_parameters():
                if name in fisher:
                    loss += torch.sum(fisher[name] * (param - theta_star[name]).pow(2))
        return lambda_ewc * loss



    def prepare_all_datasets(self):
        # TODO: Implement dataset preparation logic
        # CIFAR10 datasets
        batch_size = self.config.batch_size
        cifar10_train_dataset = datasets.CIFAR10(root=f'{self.config.root_dir}', train=True, download=True, transform=self.transform)
        cifar10_val_dataset = datasets.CIFAR10(root=f'{self.config.root_dir}', train=False, download=True, transform=self.transform)
        cifar10_train_loader = DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True)
        cifar10_val_loader = DataLoader(cifar10_val_dataset, batch_size=batch_size, shuffle=False)

        # CIFAR-100 datasets
        cifar100_train_dataset = datasets.CIFAR100(root=f'{self.config.root_dir}', train=True, download=True, transform=self.transform)
        cifar100_val_dataset = datasets.CIFAR100(root=f'{self.config.root_dir}', train=False, download=True, transform=self.transform)
        cifar100_train_loader = DataLoader(cifar100_train_dataset, batch_size=batch_size, shuffle=True)
        cifar100_val_loader = DataLoader(cifar100_val_dataset, batch_size=batch_size, shuffle=False)

        # STL-10 datasets
        stl10_train_dataset = datasets.STL10(root=f'{self.config.root_dir}', split='train', download=True, transform=self.transform)
        stl10_val_dataset = datasets.STL10(root=f'{self.config.root_dir}', split='test', download=True, transform=self.transform)
        stl10_train_loader = DataLoader(stl10_train_dataset, batch_size=batch_size, shuffle=True)
        stl10_val_loader = DataLoader(stl10_val_dataset, batch_size=batch_size, shuffle=False)

        # Caltech-256 datasets
        caltech256_dataset = datasets.Caltech256(root=f'{self.config.root_dir}/caltech256', download=True, transform=self.transform)
        total_size = len(caltech256_dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size
        caltech256_train_dataset, caltech256_val_dataset = torch.utils.data.random_split(caltech256_dataset, [train_size, val_size])
        # caltech256_train_dataset = datasets.Caltech256(root=f'{self.config.root_dir}/caltech256', download=True, transform=self.transform)
        # caltech256_val_dataset = datasets.Caltech256(root=f'{self.config.root_dir}/caltech256', download=True, transform=self.transform)
        caltech256_train_loader = DataLoader(caltech256_train_dataset, batch_size=batch_size, shuffle=True)
        caltech256_val_loader = DataLoader(caltech256_val_dataset, batch_size=batch_size, shuffle=False)

        # Tiny ImageNet datasets
        tiny_imagenet_train_dataset = datasets.ImageFolder(root=f'{self.config.root_dir}/tiny-imagenet-200/train', transform=self.transform)
        tiny_imagenet_val_dataset = datasets.ImageFolder(root=f'{self.config.root_dir}/tiny-imagenet-200/val', transform=self.transform)
        tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=batch_size, shuffle=True)
        tiny_imagenet_val_loader = DataLoader(tiny_imagenet_val_dataset, batch_size=batch_size, shuffle=False)

        self.datasets = {
            'CIFAR10': (cifar10_train_loader, cifar10_val_loader),
            'CIFAR100': (cifar100_train_loader, cifar100_val_loader),
            'STL10': (stl10_train_loader, stl10_val_loader),
            'Caltech256': (caltech256_train_loader, caltech256_val_loader),
            'TinyImageNet': (tiny_imagenet_train_loader, tiny_imagenet_val_loader),
        }

        self.order_of_tasks = ['CIFAR10', 'CIFAR100', 'STL10', 'Caltech256', 'TinyImageNet']

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
            'CIFAR10': 30,
            'CIFAR100': 30,
            'STL10': 30,
            'Caltech256': 30,
            'TinyImageNet': 30,
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

        self.writer.add_scalar( f'{task_name}/validation_loss', loss, iteration)
        self.writer.add_scalar( f'{task_name}/validation_accuracy', accuracy, iteration)


    def train_dataset(
            self,
            task_name:str,
            Q_prev = None,
            starting_step:int=0,
            epochs: int = 1,
            log_deltas: bool = False,
            mode="nostalgia",
        ):

        train_loader, _ = self.datasets[task_name]

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

                if mode == "l2sp":
                    loss += self.imageClassifier.l2sp_regularization(self.theta_0, lambda_l2sp=1e-4) #type: ignore

                elif mode == "EWC":
                    loss += self.EWC_regularization(lambda_ewc=1e-3)

                loss.backward()
                optimizer.step()

                accuracy = (output.argmax(dim=1) == target).float().mean().item()

                # Logging:
                self.writer.add_scalar(
                    f'{task_name}/train_loss',
                    loss.item(),
                    niter
                )

                self.writer.add_scalar(
                    f'{task_name}/train_accuracy',
                    accuracy,
                    niter
                )

                if (niter+1) % self.config.validate_after_steps == 0:
                    self.validate(niter, task_name)

                niter += 1

        return niter



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




        for task_name in self.order_of_tasks:
            self.finished_datasets.append(task_name)

            epochs = self.epochs_per_task[task_name]
            train_loader, _ = self.datasets[task_name]

            print(f"\n\nStarting training on {task_name} for {epochs} epochs.")

            total_steps = self.train_dataset(
                task_name=task_name,
                Q_prev=Q_prev,
                starting_step=total_steps,
                epochs=epochs,
                log_deltas=log_deltas,
                mode=self.config.mode,
            )

            if self.config.mode == "EWC":
                fisher_information = self.imageClassifier.get_fisher_information(dataloader=train_loader)
                self.store_ewc_information(task_name, fisher_information)

            print(f"Immediately after training on {task_name}, total steps: {total_steps}")

            if self.config.mode == "nostalgia":
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

                self.imageClassifier._merge_and_unload_peft()
                self.imageClassifier._apply_peft()


            # Save model after each task
            self.save_model(f'./model_weights/nostalgia_model_after_{task_name}.pth')

            print(f"Completed training on {task_name}.")
            print(f"Total steps so far: {total_steps}.")
            task_counter += 1

        print("Training completed for all tasks.")





if __name__ == "__main__": 

    args = get_args()

    config = NostalgiaConfig(
        mode=args.mode,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        hessian_eigenspace_dim=args.hessian_eigenspace_dim,
        validate_after_steps=args.validate_after_steps,
        log_dir=args.log_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )

    experiment = NostalgiaExperiment(config)
    experiment.load_model()  # Load pre-trained model if available
    experiment.train()

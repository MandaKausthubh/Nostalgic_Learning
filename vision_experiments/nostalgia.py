import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict

from utils_new.accumulate import accumulate_hessian_eigenspace
from utils_new.ortho_accumulate import update_Q_lambda_union
from utils_new.hessian import compute_Q_for_task

from datasets import get_imagenet_split

from models.vit32 import ImageClassifierViT
from tqdm import tqdm
import datetime

from torch.utils.tensorboard import SummaryWriter
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')



def get_args():
    parser = argparse.ArgumentParser(description="Nostalgia Vision Experiment")
    parser.add_argument('--mode', type=str, default='nostalgia', help='Training mode: nostalgia, l2sp, EWC, Adam',
                        choices=['nostalgia', 'l2sp', 'EWC', 'Adam'])
    parser.add_argument('--root_dir', type=str, default='/Users/mandakausthubh/data', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--batch_size_for_accumulate', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for optimizer')
    parser.add_argument('--downstream_learning_rate', type=float, default=1e-2, help='Learning rate for downstream tasks')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weighted decay rate for optimizer')
    parser.add_argument('--device', type=str, default='mps', help='Device to use for training (e.g., cpu, cuda, mps)',
                        choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--hessian_eigenspace_dim', type=int, default=32, help='Dimension of Hessian eigenspace')
    parser.add_argument('--validate_after_steps', type=int, default=100, help='Validation frequency in steps')
    parser.add_argument('--log_dir', type=str,
                        default=f'./logs/nostalgia_vision_experiment/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--ewc_lambda', type=float, default=1e-3, help='EWC regularization strength')
    parser.add_argument('--l2sp_lambda', type=float, default=1e-4, help='L2-SP regularization strength')
    parser.add_argument('--reset_lora', type=str2bool, default=False, help='Whether to reset LoRA parameters before training each task')
    parser.add_argument('--accumulate_mode', type=str, default='union', help='Mode for accumulating Hessian eigenspaces',
                        choices=['accumulate', 'union'])
    parser.add_argument('--iterations_of_accumulate', type=int, default=5, help='Number of iterations for accumulating Hessian eigenspace')
    parser.add_argument('--log_deltas', type=str2bool, default=True, help='Whether to log parameter deltas during training')
    parser.add_argument('--use_scaling', type=str2bool, default=False, help='Whether to use scaling for Hessian eigenspace')
    parser.add_argument('--adapt_downstream_tasks', type=str2bool, default=False, help='Whether to adapt downstream tasks using nostalgia method')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--head_warmup_epochs', type=int, default=2, help='Number of epochs to warm up task heads during validation')
    parser.add_argument('--base_optimizer', type=str, default='sgd', help='Base optimizer to use', choices=['sgd', 'adamw', 'adam'])
    return parser.parse_args()





@dataclass
class NostalgiaConfig:
    mode: str = 'nostalgia'
    seed: int = 42
    root_dir: str = '/Users/mandakausthubh/data'
    batch_size: int = 64
    batch_size_for_accumulation: int = 16
    learning_rate: float = 1e-4
    downstream_learning_rate: float = 1e-2
    base_optimizer: str = 'sgd'
    weight_decay: float = 1e-4
    device: str = 'mps'
    validate_after_steps: int = 10
    log_deltas: bool = True
    num_workers: int = 4

    hessian_eigenspace_dim: int = 16

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_modules: Optional[list] = None

    ewc_lambda: float = 1e-4
    l2sp_lambda: float = 1e-4
    reset_lora: bool = True
    head_warmup_epochs: int = 5

    accumulate_mode: str = 'accumulate'  # or 'union'
    iterations_of_accumulation: int = 5
    use_scaling: bool = True
    adapt_downstream_tasks: bool = False
    log_dir: str = f'./logs/nostalgia_vision_experiment/{mode}/{learning_rate}/{lora_r}/{hessian_eigenspace_dim}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'



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
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])


        using_nostalgia = (self.config.mode == "nostalgia")

        self.imageClassifier = ImageClassifierViT(
            learning_rate=self.config.learning_rate,
            lora_r=self.config.lora_r, 
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_modules=self.config.lora_modules,
            use_nostalgia=using_nostalgia,
            weight_decay=self.config.weight_decay,
            downstream_learning_rate=self.config.downstream_learning_rate,
            optimizer_type=self.config.base_optimizer,
        ).to(self.config.device)
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        self.finished_datasets = []

        self.ewc_fisher: Dict[str, torch.Tensor] = {}
        self.ewc_params: Dict[str, torch.Tensor] = {}


    def store_ewc_information(
        self,
        fisher_information: Dict[str, torch.Tensor],
        alpha: float = 0.9,
    ):
        # First task:
        if len(self.ewc_fisher) == 0:
            self.ewc_fisher = {
                k: v.detach().clone() 
                for k, v in fisher_information.items()
            }
        else:
            for k, v in fisher_information.items():
                if k in self.ewc_fisher:
                    self.ewc_fisher[k] = (
                        alpha * self.ewc_fisher[k] + (1 - alpha) * v.detach().clone()
                    )
                else:
                    self.ewc_fisher[k] = v.detach().clone()

        self.ewc_params = {
            name: p.detach().clone() 
            for name, p in self.imageClassifier.backbone.named_parameters()
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
        lambda_ewc = None,
    ):
        lambda_ewc = lambda_ewc if lambda_ewc is not None else self.config.ewc_lambda
        if not self.ewc_fisher:
            return torch.tensor(0.0, device=self.config.device)

        loss = torch.tensor(0.0, device=self.config.device)

        for name, param in self.imageClassifier.backbone.named_parameters():
            if name in self.ewc_fisher:
                loss += self.ewc_fisher[name] * (param - self.ewc_params[name]).pow(2).sum()

        return (lambda_ewc/2) * loss



    def prepare_all_datasets(self):
        # TODO: Implement dataset preparation logic
        # CIFAR10 datasets
        batch_size = self.config.batch_size
        # cifar10_train_dataset = datasets.CIFAR10(root=f'{self.config.root_dir}', train=True, download=True, transform=self.transform)
        # cifar10_val_dataset = datasets.CIFAR10(root=f'{self.config.root_dir}', train=False, download=True, transform=self.transform)
        # cifar10_train_loader = DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True)
        # cifar10_val_loader = DataLoader(cifar10_val_dataset, batch_size=batch_size, shuffle=False)
        #
        # # CIFAR-100 datasets
        # cifar100_train_dataset = datasets.CIFAR100(root=f'{self.config.root_dir}', train=True, download=True, transform=self.transform)
        # cifar100_val_dataset = datasets.CIFAR100(root=f'{self.config.root_dir}', train=False, download=True, transform=self.transform)
        # cifar100_train_loader = DataLoader(cifar100_train_dataset, batch_size=batch_size, shuffle=True)
        # cifar100_val_loader = DataLoader(cifar100_val_dataset, batch_size=batch_size, shuffle=False)
        #
        # # STL-10 datasets
        # stl10_train_dataset = datasets.STL10(root=f'{self.config.root_dir}', split='train', download=True, transform=self.transform)
        # stl10_val_dataset = datasets.STL10(root=f'{self.config.root_dir}', split='test', download=True, transform=self.transform)
        # stl10_train_loader = DataLoader(stl10_train_dataset, batch_size=batch_size, shuffle=True)
        # stl10_val_loader = DataLoader(stl10_val_dataset, batch_size=batch_size, shuffle=False)
        #
        # # Caltech-256 datasets
        # caltech256_dataset = datasets.Caltech256(root=f'{self.config.root_dir}/caltech256', download=True, transform=self.transform)
        # total_size = len(caltech256_dataset)
        # train_size = int(0.9 * total_size)
        # val_size = total_size - train_size
        # caltech256_train_dataset, caltech256_val_dataset = torch.utils.data.random_split(caltech256_dataset, [train_size, val_size])
        # # caltech256_train_dataset = datasets.Caltech256(root=f'{self.config.root_dir}/caltech256', download=True, transform=self.transform)
        # # caltech256_val_dataset = datasets.Caltech256(root=f'{self.config.root_dir}/caltech256', download=True, transform=self.transform)
        # caltech256_train_loader = DataLoader(caltech256_train_dataset, batch_size=batch_size, shuffle=True)
        # caltech256_val_loader = DataLoader(caltech256_val_dataset, batch_size=batch_size, shuffle=False)
        #
        # # Tiny ImageNet datasets
        # tiny_imagenet_train_dataset = datasets.ImageFolder(root=f'{self.config.root_dir}/tiny-imagenet-200/train', transform=self.transform)
        # tiny_imagenet_val_dataset = datasets.ImageFolder(root=f'{self.config.root_dir}/tiny-imagenet-200/val', transform=self.transform)
        # tiny_imagenet_train_loader = DataLoader(tiny_imagenet_train_dataset, batch_size=batch_size, shuffle=True)
        # tiny_imagenet_val_loader = DataLoader(tiny_imagenet_val_dataset, batch_size=batch_size, shuffle=False)
        #
        # self.datasets = {
        #     'CIFAR10': (cifar10_train_loader, cifar10_val_loader),
        #     'CIFAR100': (cifar100_train_loader, cifar100_val_loader),
        #     'STL10': (stl10_train_loader, stl10_val_loader),
        #     'Caltech256': (caltech256_train_loader, caltech256_val_loader),
        #     'TinyImageNet': (tiny_imagenet_train_loader, tiny_imagenet_val_loader),
        # }
        #
        # self.order_of_tasks = ['CIFAR10', 'CIFAR100', 'STL10', 'Caltech256', 'TinyImageNet']
        #
        # self.dataset_num_classes = {
        #     'CIFAR10': 10,
        #     'CIFAR100': 100,
        #     'STL10': 10,
        #     'Caltech256': 256,
        #     'TinyImageNet': 200,
        # }
        #
        # for task_name in self.datasets.keys():
        #     self.imageClassifier.add_task(task_name, self.dataset_num_classes[task_name])
        #
        # self.epochs_per_task = {
        #     'CIFAR10': 30,
        #     'CIFAR100': 30,
        #     'STL10': 30,
        #     'Caltech256': 30,
        #     'TinyImageNet': 30,
        # }

        # Using ImageNet variants as datasets

        dataset_splits = get_imagenet_split(
            location=self.config.root_dir,
            preprocess=self.transform,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )

        dataset_splits_for_accumulate = get_imagenet_split(
            location=self.config.root_dir,
            preprocess=self.transform,
            batch_size=self.config.batch_size_for_accumulation,
            num_workers=self.config.num_workers
        )

        self.datasets = {
            f'ImageNet/Split{i}': (dataset.train_loader, dataset.test_loader) 
            for i, dataset in enumerate(dataset_splits, start=1)
        }

        self.dataset_for_accumulate = {
            f'ImageNet/Split{i}': (dataset.train_loader, dataset.test_loader)
            for i, dataset in enumerate(dataset_splits_for_accumulate, start=1)
        }

        self.order_of_tasks = [f'ImageNet/Split{i}' for i in range(1, len(dataset_splits)+1)]
        self.dataset_num_classes = {
            task_name: 200 for task_name in (self.order_of_tasks)  # hard coding this
        }

        for task_name in self.datasets.keys():
            self.imageClassifier.add_task(task_name, self.dataset_num_classes[task_name])

        self.epochs_per_task = {
            task_name: 5 for task_name in self.datasets.keys()
        }



    def retrain_task_head(
        self,
        task_name: str,
        epochs: int = 5,
    ):
        train_loader, _ = self.datasets[task_name]
        self.imageClassifier.set_active_task(task_name)
        criterion = self.imageClassifier.criterion

        # Freeze all parameters except task head
        for param in self.imageClassifier.backbone.parameters():
            param.requires_grad = False

        self.imageClassifier.task_head_list[task_name].train()

        optimizer = torch.optim.Adam(
            self.imageClassifier.task_head_list[task_name].parameters(),
            lr=self.config.learning_rate
        )

        print("Retraining task head for", task_name)

        for epoch in range(epochs):
            step = 0
            progress_bar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch}. Retraining head for {task_name}")
            for input, target in progress_bar:
                self.imageClassifier.train()
                input, target = input.to(self.config.device), target.to(self.config.device)
                input = self.imageClassifier.preprocess_inputs(input)

                optimizer.zero_grad()
                output = self.imageClassifier(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if step % 50 == 0:
                    progress_bar.set_postfix({'loss': loss.item()})

            # Validate after each epoch
            # self.validate_dataset(val_loader, criterion, iteration=epoch, task_name=task_name)

        for param in self.imageClassifier.backbone.parameters():
            param.requires_grad = True


    def validate_dataset(
            self,
            val_loader: DataLoader,
            criterion,
            iteration: int = 0,
            task_name: Optional[str] = None
    ):
        self.imageClassifier.set_active_task(task_name)

        if self.config.adapt_downstream_tasks and task_name not in self.finished_datasets:
            print(f"Retraining task head for {task_name} before validation.")
            assert task_name is not None, "Task name must be provided for retraining task head."
            self.retrain_task_head(task_name, epochs=5)

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

        loss = total_loss / len(val_loader.dataset)     # type: ignore
        accuracy = accuracy / len(val_loader.dataset)   # type: ignore

        self.writer.add_scalar( f'{task_name}/validation_loss', loss, iteration)
        self.writer.add_scalar( f'{task_name}/validation_accuracy', accuracy, iteration)


    def train_dataset(
            self,
            task_name:str,
            Q_prev = None,
            scaling = None,
            starting_step:int=0,
            epochs: int = 1,
            log_deltas: bool = False,
            mode="nostalgia",
        ):

        train_loader, _ = self.datasets[task_name]

        self.imageClassifier.set_active_task(task_name)

        print("Scaling shape:", None if scaling is None else scaling.shape)
        self.imageClassifier.set_Q(Q_prev, scaling)
        criterion = self.imageClassifier.criterion
        niter = starting_step

        self.retrain_task_head(task_name, epochs=self.config.head_warmup_epochs)

        optimizer = self.imageClassifier.configure_optimizers(
            writter=self.writer if log_deltas else None,
            iteration=starting_step
        )

        for epoch in range(epochs):
            for input, target in tqdm((train_loader), desc=f"{epoch}. Training on {task_name}", ncols=80):
                self.imageClassifier.train()
                input, target = input.to(self.config.device), target.to(self.config.device)
                input = self.imageClassifier.preprocess_inputs(input)

                optimizer.zero_grad()
                output = self.imageClassifier(input)
                loss = criterion(output, target)

                if mode == "l2sp":
                    loss += self.imageClassifier.l2sp_regularization(self.theta_0, lambda_l2sp=self.config.l2sp_lambda)

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

                if (niter+1) % 500 == 0:
                    torch.cuda.synchronize() if self.config.device == 'cuda' else None

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


    def train(self, erase_past=False):
        self.prepare_all_datasets()
        torch.cuda.empty_cache()

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

            print(f"Using scaling: {self.config.use_scaling}")
            total_steps = self.train_dataset(
                task_name=task_name,
                Q_prev=Q_prev,
                scaling=Lambda_prev if self.config.use_scaling else None,   # TODO: Check this
                starting_step=total_steps,
                epochs=epochs,
                log_deltas=self.config.log_deltas,
                mode=self.config.mode,
            )

            if self.config.mode == "EWC":
                fisher_information = self.imageClassifier.get_fisher_information(dataloader=train_loader)
                self.store_ewc_information(fisher_information)

            print(f"Immediately after training on {task_name}, total steps: {total_steps}")

            if self.config.mode == "nostalgia":
                # After training on the current task, compute the Hessian eigenspace
                if self.config.device == 'cuda':
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=False, enable_mem_efficient=False, enable_math=True
                    ):
                        self.imageClassifier.set_active_task(task_name)
                        Q_curr, Lambda_curr = None, None

                        assert self.config.iterations_of_accumulation >= 1, "iterations_of_accumulation must be at least 1"

                        for _ in range(self.config.iterations_of_accumulation):
                            Q_new, Lambda_new = compute_Q_for_task(
                                model=self.imageClassifier,
                                train_loader=self.dataset_for_accumulate[task_name][0],
                                device=self.config.device,
                                k=self.config.hessian_eigenspace_dim
                            )

                            if self.config.accumulate_mode == 'union':
                                print("Updating Hessian eigenspace using union method.")
                                Q_curr, Lambda_curr = update_Q_lambda_union(
                                    Q_curr, Lambda_curr,
                                    Q_new, Lambda_new,
                                    k_max=self.config.hessian_eigenspace_dim * 20
                                )
                            elif self.config.accumulate_mode == 'accumulate':
                                Q_prev, Lambda_prev = accumulate_hessian_eigenspace(
                                    Q_curr, Lambda_curr,
                                    Q_new, Lambda_new,
                                    t=task_counter, k=self.config.hessian_eigenspace_dim
                                )
                            else:
                                raise ValueError(f"Unknown accumulate_mode: {self.config.accumulate_mode}")

                        assert Q_curr is not None and Lambda_curr is not None, "Q_curr and Lambda_curr should not be None after accumulation."

                        if self.config.accumulate_mode == 'union':
                            print("Updating Hessian eigenspace using union method.")
                            Q_prev, Lambda_prev = update_Q_lambda_union(
                                Q_prev, Lambda_prev,
                                Q_curr, Lambda_curr,
                                k_max=self.config.hessian_eigenspace_dim * 20
                            )
                        elif self.config.accumulate_mode == 'accumulate':
                            Q_prev, Lambda_prev = accumulate_hessian_eigenspace(
                                Q_prev, Lambda_prev,
                                Q_curr, Lambda_curr,
                                t=task_counter, k=self.config.hessian_eigenspace_dim
                            )
                        else:
                            raise ValueError(f"Unknown accumulate_mode: {self.config.accumulate_mode}")
                else:
                    Q_curr, Lambda_curr = compute_Q_for_task(
                        model=self.imageClassifier,
                        train_loader=self.dataset_for_accumulate[task_name][0],
                        device=self.config.device,
                        k=self.config.hessian_eigenspace_dim
                    )

                    if self.config.accumulate_mode == 'union':
                        print("Updating Hessian eigenspace using union method.")
                        Q_prev, Lambda_prev = update_Q_lambda_union(
                            Q_prev, Lambda_prev,
                            Q_curr, Lambda_curr,
                            k_max=self.config.hessian_eigenspace_dim * 20
                        )
                    elif self.config.accumulate_mode == 'accumulate':
                        Q_prev, Lambda_prev = accumulate_hessian_eigenspace(
                            Q_prev, Lambda_prev,
                            Q_curr, Lambda_curr,
                            t=task_counter, k=self.config.hessian_eigenspace_dim
                        )
                    else:
                        raise ValueError(f"Unknown accumulate_mode: {self.config.accumulate_mode}")

                if self.config.reset_lora:
                    self.imageClassifier._merge_and_unload_peft()
                    self.imageClassifier._apply_peft()

            # Save model after each task
            self.save_model(f'./model_weights/vision/nostalgia_model_after_{task_name}.pth')

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
        batch_size_for_accumulation=args.batch_size_for_accumulate,
        learning_rate=args.learning_rate,
        downstream_learning_rate=args.downstream_learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        hessian_eigenspace_dim=args.hessian_eigenspace_dim,
        validate_after_steps=args.validate_after_steps,
        # log_dir=args.log_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        ewc_lambda=args.ewc_lambda,
        l2sp_lambda=args.l2sp_lambda,
        reset_lora=args.reset_lora,
        accumulate_mode=args.accumulate_mode,
        iterations_of_accumulation=args.iterations_of_accumulate,
        log_deltas=args.log_deltas,
        use_scaling=args.use_scaling,
        adapt_downstream_tasks=args.adapt_downstream_tasks,
        num_workers=args.num_workers,
        head_warmup_epochs=args.head_warmup_epochs,
        base_optimizer=args.base_optimizer,
    )

    config.log_dir = f'./logs/nostalgia_vision_experiment/{config.mode}/{config.learning_rate}/{config.lora_r}/{config.hessian_eigenspace_dim}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'



    print("Experiment Configuration:")
    for field in config.__dataclass_fields__:
        print(f"{field}: {getattr(config, field)}")

    experiment = NostalgiaExperiment(config)
    experiment.load_model()  # Load pre-trained model if available

    # Create a txt file to save the config:
    with open(f'{config.log_dir}/config.txt', 'w') as f:
        for field in config.__dataclass_fields__:
            f.write(f"{field}: {getattr(config, field)}\n")

    experiment.train()

    # Testing merge and unload/apply PEFT
    # experiment.imageClassifier._merge_and_unload_peft()
    # experiment.imageClassifier._apply_peft()

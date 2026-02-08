from dataclasses import dataclass
import datetime
from typing import Optional

import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm.gui import tqdm

from models.vit32 import ImageClassifierViT
from datasets import get_imagenet_split
from utils_new.accumulate import accumulate_hessian_eigenspace
from utils_new.hessian import compute_Q_for_task
from utils_new.ortho_accumulate import update_Q_lambda_union

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
    parser = argparse.ArgumentParser(description="Nostalgia Experiment Configuration")
    parser.add_argument('--root_dir', type=str, default='~/data', help='Root directory for datasets')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')
    parser.add_argument('--lora_rank', type=int, default=16, help='Rank for LoRA adaptation')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha for LoRA adaptation')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout rate for LoRA adaptation')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer type (e.g., "adamw")')
    parser.add_argument('--hessian_dim', type=int, default=32, help='Dimensionality of the Hessian eigenspace')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--downstream_learning_rate', type=float, default=3e-3, help='Learning rate for downstream task head training')
    parser.add_argument('--log_deltas', type=str2bool, default=True, help='Whether to log gradient deltas in TensorBoard')
    parser.add_argument('--use_scaling', type=str2bool, default=False, help='Whether to use eigenvalue-aware scaling in the optimizer')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train each task')
    parser.add_argument('--validate_every', type=int, default=100, help='Number of steps between validations')
    parser.add_argument('--hessian_average_epochs', type=int, default=5, help='Number of epochs to average for Hessian computation')
    parser.add_argument('--num_warmup_epochs', type=int, default=10, help='Number of warmup epochs before applying nostalgia')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--batch_size_for_accumulate', type=int, default=16, help='Batch size to use when computing Hessian eigenspace for accumulation')
    parser.add_argument('--accumulate_mode', type=str, default='accumulate', choices=['accumulate', 'union'], help='Mode for accumulating Hessian eigenspaces ("accumulate" or "union")')
    parser.add_argument('--merge_mode', type=str, default='union', choices=['accumulate', 'union'], help='Mode for merging Hessian eigenspaces ("accumulate" or "union")')




@dataclass
class NostalgiaConfig:
    mode: str = "nostalgia"
    seed: int = 42
    dataset_dir: str = "~/data"

    # Model specifications
    device: str = "cuda"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Optimizer specifications
    optimizer: str = "adamw"
    hessian_dim: int = 32
    learning_rate: float = 1e-5
    downstream_learning_rate: float = 3e-3
    log_deltas: bool = True
    use_scaling: bool = False

    # Training dyanmics
    num_epochs: int = 20
    validate_every: int = 100
    hessian_average_epochs: int = 5
    num_warmup_epochs: int = 10
    num_workers: int = 4
    batch_size: int = 512
    batch_size_for_accumulate: int = 16
    accumulate_mode: str = "accumulate" # or "union"
    merge_mode: str = "union" # or "accumulate"
    log_dir: str = f'./logs/nostalgia_vision_experiment/{mode}/{learning_rate}/{lora_rank}/{hessian_dim}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'




class NostalgiaExperiment:
    def __init__(self, config: NostalgiaConfig):
        self.config = config
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
        self.using_nostalgia = (self.config.mode == "nostalgia")
        self.writter = SummaryWriter(log_dir=self.config.log_dir)
        self.global_step = 0

        self.setup_model()
        self.prepare_data()

    def setup_model(self):
        self.model = ImageClassifierViT(
            learning_rate=self.config.learning_rate,
            downstream_learning_rate=self.config.downstream_learning_rate,
            lora_r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            use_nostalgia=self.using_nostalgia,
            lora_modules=None,  # Use default LoRA modules for ViT
            weight_decay=0.0,
            optimizer_type=self.config.optimizer
        )

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def prepare_data(self):
        dataset_splits = get_imagenet_split(
            self.config.dataset_dir,
            self.transform,
            self.config.batch_size,
            self.config.num_workers
        )

        datasets_splits_for_accumulate = get_imagenet_split(
            self.config.dataset_dir,
            self.transform,
            self.config.batch_size,
            self.config.num_workers
        )

        self.datasets = {
            f'ImageNet-Split-{i}': (datasets.train_loader, datasets.test_loader)
            for i, datasets in enumerate(dataset_splits, start=1)
        }

        self.datasets_accumulate = {
            f'ImageNet-Split-{i}': (datasets.train_loader, datasets.test_loader)
            for i, datasets in enumerate(datasets_splits_for_accumulate, start=1)
        }

        self.order_of_tasks = [
            f'ImageNet-Split-{i}' 
            for i in range(1, len(dataset_splits) + 1)
        ]

        for task_name in self.datasets.keys():
            self.model.add_task(task_name, 200) # Hardcoded 200 classes per split

        self.finished_datasets = []

    def train_taskhead(self, task_name: str, epochs: int = 5):
        train_loader, _ = self.datasets[task_name]
        self.model.set_active_task(task_name)
        criterion = self.model.criterion

        self.model.task_head_list[task_name].train()
        self.model.backbone.eval()

        optimizer = torch.optim.AdamW(
            self.model.task_head_list[task_name].parameters(),
            lr=self.config.downstream_learning_rate,
            weight_decay=0.0
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * epochs,
            eta_min=1e-6
        )

        print(f"Training task head for {task_name}...")
        for epoch in range(epochs):
            step = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
                inputs = self.model.preprocess_inputs(inputs)  # type: ignore

                optimizer.zero_grad()
                with torch.no_grad():
                    features = self.model.backbone(inputs)
                outputs = self.model.task_head_list[task_name](features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % 10 == 0:
                    progress_bar.set_postfix({"loss": loss.item()})
                step += 1

    def validate_for_dataset(
        self,
        task_name: str,
    ):
        _, test_loader = self.datasets[task_name]
        self.model.set_active_task(task_name)
        self.model.eval()

        correct = 0
        loss = 0.0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Validating {task_name}"):
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
                inputs = self.model.preprocess_inputs(inputs)  # type: ignore
                features = self.model.backbone(inputs)
                outputs = self.model.task_head_list[task_name](features)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                loss += self.model.criterion(outputs, targets).item() * targets.size(0)
                correct += (predicted == targets).sum().item()

        self.writter.add_scalar(f'{task_name}/Loss', loss / total, self.global_step)
        self.writter.add_scalar(f'{task_name}/Accuracy', correct / total, self.global_step)

    def validate_all_datasets(self, current_task: Optional[str] = None):
        for task_name in self.finished_datasets + ([current_task] if current_task else []):
            self.validate_for_dataset(task_name)

    def train_for_dataset(
        self,
        task_name: str,
        Q_prev: Optional[torch.Tensor] = None,
        Lamda_prev: Optional[torch.Tensor] = None,
    ):
        train_loader, _ = self.datasets[task_name]
        self.model.set_active_task(task_name)

        criterion = self.model.criterion
        self.model.set_Q(Q_prev, scaling=Lamda_prev)

        optimizer = self.model.configure_optimizers(
            writter=self.writter,
            iteration=self.global_step
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer,   # type: ignore
            T_max=len(train_loader) * self.config.num_epochs,
            eta_min=1e-6
        )

        for epoch in range(self.config.num_epochs):
            step = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
                inputs = self.model.preprocess_inputs(inputs)  # type: ignore

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

                self.writter.add_scalar(
                    f'{task_name}/train_loss',
                    loss.item(), self.global_step
                )

                self.writter.add_scalar(
                    f'{task_name}/Learning_Rate',
                    scheduler.get_last_lr()[0], self.global_step
                )

                self.writter.add_scalar(
                    f'{task_name}/train_accuracy',
                    (outputs.argmax(dim=1) == targets).float().mean().item(),
                )

                if step % 10 == 0:
                    progress_bar.set_postfix({"loss": loss.item()})

                if self.global_step % self.config.validate_every == 0:
                    self.validate_all_datasets(task_name)

                step += 1
                self.global_step += 1

                if epoch+1 % self.config.hessian_average_epochs == 0 and epoch!=0:
                    Q, Lambda = None, None
                    for i, task in enumerate(self.finished_datasets):
                        Q_task, Lambda_task = self.get_Q_and_Lambda(
                            task, mode=self.config.accumulate_mode
                        )

                        assert Q_task is not None, f"Failed to compute Q for task {task}"
                        assert Lambda_task is not None, f"Failed to compute Lambda for task {task}"

                        if self.config.merge_mode == "accumulate":
                            Q, Lambda = accumulate_hessian_eigenspace(
                                Q_old=Q, Lambda_old=Lambda,
                                Q_new=Q_task, Lambda_new=Lambda_task,
                                t=(i+1), k=self.config.hessian_dim,
                            )
                        elif self.config.merge_mode == "union":
                            Q, Lambda = update_Q_lambda_union(
                                Q_union=Q, lambda_union=Lambda,
                                Q_new=Q_task, lambda_new=Lambda_task,
                                k_max=self.config.hessian_dim,
                            )
                        else:
                            raise ValueError(f"Invalid merge_mode: {self.config.merge_mode}")

                    if self.config.use_scaling:
                        optimizer.set_Q(Q, scaling=Lambda)  # type: ignore
                    else:
                        optimizer.set_Q(Q, scaling=None)  # type: ignore


    def get_Q_and_Lambda(
        self,
        task_name: str,
        mode: str = "accumulate"
    ):
        Q, Lambda = None, None
        train_loader, _ = self.datasets_accumulate[task_name]

        for i in range(self.config.hessian_average_epochs):
            Q_new, Lambda_new = compute_Q_for_task(
                model=self.model,
                device=self.model.device,
                k = self.config.hessian_dim,
                train_loader=train_loader,
            )

            if mode == "accumulate":
                Q, Lambda = accumulate_hessian_eigenspace(
                    Q_old=Q, Lambda_old=Lambda,
                    Q_new=Q_new, Lambda_new=Lambda_new,
                    t=(i+1), k=self.config.hessian_dim,
                )
            elif mode == "union":
                Q, Lambda = update_Q_lambda_union(
                    Q_union=Q, lambda_union=Lambda,
                    Q_new=Q_new, lambda_new=Lambda_new,
                    k_max=self.config.hessian_dim,
                )
            else:
                raise ValueError(f"Invalid accumulate_mode: {self.config.accumulate_mode}")

        return Q, Lambda

    def train(self):
        # Initial version is only for experimentation
        # TODO: Create separate code for other modes
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        Q_prev, Lambda_prev = None, None
        for task_name in self.order_of_tasks:
            self.train_for_dataset(
                task_name,
                Q_prev=Q_prev,
                Lamda_prev=Lambda_prev,
            )

            if self.using_nostalgia:
                Q_prev, Lambda_prev = self.get_Q_and_Lambda(
                    task_name, mode=self.config.accumulate_mode
                )

            self.finished_datasets.append(task_name)




if __name__ == "__main__":
    args = get_args()
    config = NostalgiaConfig(**vars(args))
    experiment = NostalgiaExperiment(config)
    experiment.train()


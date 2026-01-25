from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

from peft import get_peft_model, LoraConfig

from utils_new.nostalgia import NostalgiaOptimizer

from transformers import AutoModel, AutoConfig, AutoProcessor



class ViTClassifier(nn.Module):
    def __init__(self):
        super(ViTClassifier, self).__init__()
        model_id = "google/vit-base-patch32-224-in21k"
        config = AutoConfig.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.vit = AutoModel.from_pretrained(model_id, config=config)

    def forward(self, pixel_values):
        # Extract the CLS token representation
        outputs = self.vit(pixel_values=pixel_values)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        return cls_representation



# Note that your image has to be 32x32 for this model
class ImageClassifierViT(pl.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.0,
        downstream_learning_rate=0.01,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        use_peft=True,
        lora_modules=None,
        use_nostalgia=True,
        optimizer_type="adamw",
    ):
        super(ImageClassifierViT, self).__init__()
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules if lora_modules is not None else ["query", "value", "key"],
            lora_dropout=lora_dropout,
            bias="none",
            # task_type="FEATURE_EXTRACTION",
        )
        self.backbone = ViTClassifier()
        self.rep_dim = 768  # ViT base model representation dimension
        self.use_preocessor = True
        self.optimizer_type = optimizer_type

        # PEFT setup
        self.use_peft = use_peft
        self.is_peft_on = False
        if self.use_peft:
            self._apply_peft()

        # Task heads
        self.task_head_list = torch.nn.ModuleDict()
        self.active_task = None

        # Forgetting tracking
        self.track_forgetting = True
        self.previous_task_datasets : Dict[str, Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]] = {}

        # Nostalgia setup
        self.use_nostalgia = use_nostalgia
        self.nostalgia_Q: Optional[torch.Tensor] = None
        self.nostalgia_scaling: Optional[torch.Tensor] = None

        # Loss and optimizer params
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.downstream_learning_rate = downstream_learning_rate

    def preprocess_inputs(self, x):
        if self.use_preocessor:
            x = self.backbone.processor(  # type: ignore
                images=x,
                return_tensors="pt"
            )["pixel_values"].to(x.device)
        return x

    def _apply_peft(self):
        if self.use_peft:
            self.backbone = get_peft_model(self.backbone, self.lora_config) #type: ignore
            self.is_peft_on = True

    def _merge_and_unload_peft(self):
        if self.is_peft_on:
            self.backbone = self.backbone.merge_and_unload() #type: ignore

            if hasattr(self.backbone, "peft_config"):
                print("Deleting peft_config attribute...")
                delattr(self.backbone, "peft_config")

            self.is_peft_on = False



    # -------------------------------------------------------
    def append_previous_task_dataset(self, task_name, dataloaders):
        self.previous_task_datasets[task_name] = dataloaders

    def add_task(self, task_name, num_classes):
        self.task_head_list[task_name] = nn.Linear(self.rep_dim, num_classes).to(self.device)
        nn.init.trunc_normal_(self.task_head_list[task_name].weight, std=0.02)  # type: ignore
        nn.init.zeros_(self.task_head_list[task_name].bias)                     # type: ignore

        for param in self.task_head_list[task_name].parameters():
            param.requires_grad = False

    def set_active_task(self, task_name):
        if self.active_task is not None:
            for param in self.task_head_list[self.active_task].parameters():
                param.requires_grad = False

        self.active_task = task_name
        for param in self.task_head_list[task_name].parameters():
            param.requires_grad = True

    @torch.no_grad()
    def loss_for_previous_tasks(
        self,
        previous_task_datasets: Dict[str, Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]
    ):
        val_loss_dicts = {}
        train_loss_dicts = {}

        for task_name, (_, val_loader) in previous_task_datasets.items():
            self.set_active_task(task_name)
            total_loss = 0.0
            total_samples = 0

            for inputs, targets in val_loader:
                logits = self.forward(inputs)
                loss = self.criterion(logits, targets)
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            val_loss_dicts[task_name] = avg_loss

        for task_name, (train_loader, _) in previous_task_datasets.items():
            self.set_active_task(task_name)
            total_loss = 0.0
            total_samples = 0

            for inputs, targets in train_loader:
                logits = self.forward(inputs)
                loss = self.criterion(logits, targets)
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            train_loss_dicts[task_name] = avg_loss

        if val_loss_dicts == {} or train_loss_dicts == {}:
            val_loss_dicts["none"] = 0.0
            train_loss_dicts["none"] = 0.0

        return train_loss_dicts, val_loss_dicts


    # -------------------------------------------------------
    def forward(self, x):
        # print("Input shape:", x.shape)
        features = self.backbone(x)
        if self.active_task is None:
            raise ValueError("Active task is not set.")
        logits = self.task_head_list[self.active_task](features)
        return logits


    # -------------------------------------------------------
    def training_step(self, batch, _):
        inputs, targets = batch
        inputs = self.preprocess_inputs(inputs)
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        logging_dict = {"current_loss": loss.item()}
        train_loss_dicts, val_loss_dicts = self.loss_for_previous_tasks(self.previous_task_datasets)

        for task_name, val_loss in val_loss_dicts.items():
            logging_dict[f'val_loss_{task_name}'] = val_loss

        for task_name, train_loss in train_loss_dicts.items():
            logging_dict[f'train_loss_{task_name}'] = train_loss
        self.log_dict(logging_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss



    # -------------------------------------------------------
    def set_Q(self, Q: Optional[torch.Tensor], scaling: Optional[torch.Tensor] = None):
        self.nostalgia_Q = Q
        self.nostalgia_scaling = scaling

    def get_backbone_params(self):
        return [
            p for _, p in self.backbone.named_parameters()
            if p.requires_grad
        ]

    def configure_optimizers(
            self,
            writter: Optional[SummaryWriter] = None,
            iteration: int = 0,
    ):
        # Backbone params (shared, projected)
        backbone_params = self.get_backbone_params()

        # ALL head params (we will freeze/unfreeze via requires_grad)
        head_params = []
        for head in self.task_head_list.values():
            head_params.extend(
                p for p in head.parameters() if p.requires_grad
            )

        if self.optimizer_type == "sgd":
            base_optimizer = torch.optim.SGD(
                [
                    {"params": backbone_params, "lr": self.learning_rate},
                    {"params": head_params, "lr": self.downstream_learning_rate},
                ],
                momentum=0.9,
                weight_decay=0.0,
                nesterov=True,
            )
        elif self.optimizer_type == "adamw":
            base_optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": self.learning_rate},
                    {"params": head_params, "lr": self.downstream_learning_rate},
                ],
                weight_decay=0.0,
            )
        elif self.optimizer_type == "adam":
            base_optimizer = torch.optim.Adam(
                [
                    {"params": backbone_params, "lr": self.learning_rate},
                    {"params": head_params, "lr": self.downstream_learning_rate},
                ],
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")


        # base_optimizer = torch.optim.SGD(
        #     [
        #         {"params": backbone_params},
        #         {"params": head_params},
        #     ],
        #     lr=self.learning_rate,
        #     momentum=0.9,
        #     weight_decay=1e-3,
        # )

        if not self.use_nostalgia:
            return base_optimizer

        assert backbone_params and len(backbone_params) > 0, "No backbone parameters to optimize for Nostalgia."

        nostalgia_opt = NostalgiaOptimizer(
            params=backbone_params,
            base_optimizer=base_optimizer,
            device=self.device,
            dtype=backbone_params[0].dtype,
            writter=writter,
            starting_step=iteration,
            weight_decay=self.weight_decay,
        )

        if self.nostalgia_Q is not None:
            nostalgia_opt.set_Q(self.nostalgia_Q, scaling=self.nostalgia_scaling)

        return nostalgia_opt


    # -------------------------------------------------------
    def l2sp_regularization(self, model_before: Dict[str, torch.Tensor], lambda_l2sp: float):
        l2sp_loss = 0.0
        for name, param in self.backbone.named_parameters():
            if name in model_before and param.requires_grad:
                l2sp_loss += torch.sum((param - model_before[name].to(param.device)) ** 2)
        l2sp_loss = lambda_l2sp * l2sp_loss
        return l2sp_loss

    def get_fisher_information(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        fisher_information = {}
        self.backbone.eval()

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.zero_grad()
            inputs = self.preprocess_inputs(inputs)
            logits = self.forward(inputs)
            # loss = self.criterion(logits, targets)
            # loss.backward()

            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                sampled_classes = torch.multinomial(probs, num_samples=1).squeeze()

            loss = F.nll_loss(F.log_softmax(logits, dim=1), sampled_classes)
            loss.backward()

            for name, param in self.backbone.named_parameters():
                if param.grad is not None:
                    if name not in fisher_information:
                        fisher_information[name] = param.grad.data.clone().pow(2)
                    else:
                        fisher_information[name] += param.grad.data.clone().pow(2)

        # Average the Fisher Information
        num_batches = len(dataloader) #type: ignore
        for name in fisher_information:
            fisher_information[name] /= num_batches

        return fisher_information




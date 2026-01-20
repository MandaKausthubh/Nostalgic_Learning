
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModel, AutoConfig, AutoTokenizer

from peft import get_peft_model, LoraConfig
from utils_new.nostalgia import NostalgiaOptimizer








class LlamaEncoder(nn.Module):
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B", using_mps=False):
        super().__init__()
        self.model_id = model_id

        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            padding_side="right"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llama = AutoModel.from_pretrained(
            model_id,
            config=self.config,
            torch_dtype=torch.bfloat16 if not using_mps else torch.float32,
        )

        self.hidden_size = self.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a single vector per sequence
        Shape: (batch_size, hidden_dim)
        """
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Last hidden state: (B, T, D)
        hidden_states = outputs.last_hidden_state

        # Get last non-padding token index per sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)

        pooled = hidden_states[batch_indices, seq_lengths]
        return pooled




class NLPClassifierLLaMA(pl.LightningModule):
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-3B",
        learning_rate=2e-5,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        use_peft=True,
        lora_modules=None,
        use_nostalgia=True,
        using_mps=False,
    ):
        super().__init__()

        # Backbone
        self.backbone = LlamaEncoder(model_id, using_mps=using_mps)
        self.rep_dim = self.backbone.hidden_size

        # LoRA
        self.use_peft = use_peft
        self.is_peft_on = False
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules
            if lora_modules is not None
            else ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        if self.use_peft:
            self._apply_peft()

        # Task heads
        self.task_head_list = nn.ModuleDict()
        self.task_type: Dict[str, str] = {}  # "sequence" | "token"
        self.active_task: Optional[str] = None

        # Forgetting tracking
        self.previous_task_datasets: Dict[
            str, Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        ] = {}

        # Nostalgia
        self.use_nostalgia = use_nostalgia
        self.nostalgia_Q: Optional[torch.Tensor] = None
        self.nostalgia_scaling: Optional[torch.Tensor] = None

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.learning_rate = learning_rate

        self.nostalgia_Q = None
        self.nostalgia_scaling = None



    def _apply_peft(self):
        self.backbone = get_peft_model(self.backbone, self.lora_config) # type: ignore
        self.is_peft_on = True

    def _merge_and_unload_peft(self):
        if self.is_peft_on:
            self.backbone = self.backbone.merge_and_unload()   # type: ignore
            if hasattr(self.backbone, "peft_config"):
                delattr(self.backbone, "peft_config")
            self.is_peft_on = False



    def add_task(self, task_name: str, num_classes: int, task_type: str):
        """
        task_type: "sequence" or "token"
        """
        assert task_type in {"sequence", "token"}
        self.task_type[task_name] = task_type

        if task_type == "sequence":
            head = nn.Linear(self.rep_dim, num_classes)
        else:
            head = nn.Linear(self.rep_dim, num_classes)

        self.task_head_list[task_name] = head.to(self.device)

        for p in head.parameters():
            p.requires_grad = False

    def preprocess_inputs(self, x): return x


    def set_active_task(self, task_name: str):
        if self.active_task is not None:
            for p in self.task_head_list[self.active_task].parameters():
                p.requires_grad = False

        self.active_task = task_name
        for p in self.task_head_list[task_name].parameters():
            p.requires_grad = True


    def forward(self, input_ids, attention_mask, labels=None):
        if self.active_task is None:
            raise ValueError("Active task not set")

        task_type = self.task_type[self.active_task]
        outputs = self.backbone.llama(                        # type: ignore
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # (B, T, D)
        head = self.task_head_list[self.active_task]

        if task_type == "sequence":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_idx, seq_lengths]
            logits = head(pooled)
            return logits

        else:  # token-level
            logits = head(hidden_states)  # (B, T, C)
            return logits


    def training_step(self, batch, _):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)

        if self.task_type[self.active_task] == "token":  # type: ignore
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        else:
            loss = self.criterion(logits, labels)

        self.log("current_loss", loss, prog_bar=True)
        return loss


    def get_backbone_params(self):
        return [p for _, p in self.backbone.named_parameters() if p.requires_grad]


    def set_Q(self, Q: Optional[torch.Tensor], scaling: Optional[torch.Tensor] = None):
        self.nostalgia_Q = Q
        self.nostalgia_scaling = scaling


    def configure_optimizers(
        self,
        writter: Optional[SummaryWriter] = None,
        iteration: int = 0,
    ):
        backbone_params = self.get_backbone_params()
        head_params = []

        for head in self.task_head_list.values():
            head_params.extend(p for p in head.parameters() if p.requires_grad)

        base_optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params},
                {"params": head_params},
            ],
            lr=self.learning_rate,
        )

        if not self.use_nostalgia:
            return base_optimizer

        nostalgia_opt = NostalgiaOptimizer(
            params=backbone_params,
            base_optimizer=base_optimizer,
            device=self.device,
            dtype=backbone_params[0].dtype,
            writter=writter,
            starting_step=iteration,
        )

        if self.nostalgia_Q is not None:
            nostalgia_opt.set_Q(self.nostalgia_Q, self.nostalgia_scaling)

        return nostalgia_opt

    def l2sp_regularization(self, model_before: Dict[str, torch.Tensor], lambda_l2sp: float):
        l2sp_loss = 0.0
        for name, param in self.backbone.named_parameters():
            if name in model_before and param.requires_grad:
                l2sp_loss += torch.sum((param - model_before[name].to(param.device)) ** 2)
        l2sp_loss = lambda_l2sp * l2sp_loss
        return l2sp_loss

    def get_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute empirical Fisher Information for EWC.
        
        IMPORTANT:
        - Only valid for SEQUENCE-level tasks (SST-2, MNLI).
        - Token-level tasks (e.g. CoNLL) are intentionally skipped.
        - Fisher is computed ONLY over trainable (LoRA) parameters.
        """

        # -------------------------------
        # Guard: skip token-level tasks
        # -------------------------------
        if self.task_type[self.active_task] == "token":  # type: ignore
            print(
                f"[EWC] Skipping Fisher computation for token-level task: {self.active_task}"
            )
            return {}

        fisher_information: Dict[str, torch.Tensor] = {}
        self.backbone.eval()

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            self.zero_grad()

            # Forward pass
            logits = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )  # (B, C)

            # Sample labels from model distribution (empirical Fisher)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                sampled_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Negative log-likelihood
            loss = F.nll_loss(
                torch.log_softmax(logits, dim=-1),
                sampled_labels,
            )

            # Backward pass
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.backbone.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in fisher_information:
                        fisher_information[name] = param.grad.detach().clone().pow(2)
                    else:
                        fisher_information[name] += param.grad.detach().clone().pow(2)

        # Normalize
        num_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
        for name in fisher_information:
            fisher_information[name] /= num_batches

        return fisher_information

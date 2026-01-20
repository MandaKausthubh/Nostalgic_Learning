import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict

from datasets import load_dataset
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding

from utils_new.accumulate import accumulate_hessian_eigenspace
from utils_new.ortho_accumulate import update_Q_lambda_union
from utils_new.hessian import compute_Q_for_task

from models.llama3b import NLPClassifierLLaMA

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
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for optimizer')
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
    parser.add_argument('--nostalgia_dimension', type=int, default=16, help='Dimension of Hessian low-rank for nostalgia method')
    parser.add_argument('--ewc_lambda', type=float, default=1e-3, help='EWC regularization strength')
    parser.add_argument('--l2sp_lambda', type=float, default=1e-4, help='L2-SP regularization strength')
    parser.add_argument('--reset_lora', type=str2bool, default=False, help='Whether to reset LoRA parameters before training each task')
    parser.add_argument('--accumulate_mode', type=str, default='union', help='Mode for accumulating Hessian eigenspaces',
                        choices=['accumulate', 'union'])
    parser.add_argument('--log_deltas', type=str2bool, default=True, help='Whether to log parameter deltas during training')
    parser.add_argument('--use_scaling', type=str2bool, default=False, help='Whether to use scaling for Hessian eigenspace')
    parser.add_argument('--adapt_downstream_tasks', type=str2bool, default=False, help='Whether to adapt downstream tasks using nostalgia method')
    return parser.parse_args()


@dataclass
class NostalgiaConfig:
    mode: str = 'nostalgia'
    seed: int = 42
    root_dir: str = '/Users/mandakausthubh/data'
    batch_size: int = 64
    learning_rate: float = 1e-4
    device: str = 'mps'
    validate_after_steps: int = 10
    log_deltas: bool = True

    hessian_eigenspace_dim: int = 16

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_modules: Optional[list] = None

    ewc_lambda: float = 1e-4
    l2sp_lambda: float = 1e-4
    reset_lora: bool = True

    accumulate_mode: str = 'accumulate'  # or 'union'
    use_scaling: bool = True
    adapt_downstream_tasks: bool = False
    log_dir: str = f'./logs/nostalgia_nlp_experiment/{mode}/{learning_rate}/{lora_r}/{hessian_eigenspace_dim}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'




def tokenize_sst2(ex, tokenizer):
    return tokenizer(ex["sentence"], truncation=True)

def tokenize_mnli(ex, tokenizer):
    return tokenizer(ex["premise"], ex["hypothesis"], truncation=True)

def tokenize_conll(ex, tokenizer, label2id):
    tokens = tokenizer(
        ex["tokens"],
        is_split_into_words=True,
        truncation=True,
    )

    word_ids = tokens.word_ids()
    labels = []
    prev = None
    for w in word_ids:
        if w is None:
            labels.append(-100)
        elif w != prev:
            labels.append(label2id[ex["ner_tags"][w]])
        else:
            labels.append(-100)
        prev = w

    tokens["labels"] = labels
    return tokens





class NostalgiaExperiment:
    def __init__(self, config: NostalgiaConfig):
        self.config = config
        torch.manual_seed(self.config.seed)

        using_nostalgia = (self.config.mode == "nostalgia")

        self.model = NLPClassifierLLaMA(
            learning_rate=self.config.learning_rate,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_modules=self.config.lora_modules,
            use_nostalgia=using_nostalgia,
            using_mps= (self.config.device == 'mps'),
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
            for name, p in self.model.backbone.named_parameters()
            if p.requires_grad
        }


    def load_model(self, path: Optional[str] = None):
        if path is not None:
            self.model.load_state_dict(
                torch.load(path, map_location=self.config.device)
            )
        else:
            print("No pre-trained model path provided. Using random initialization.")
            NLPClassifierLLaMA().to(self.config.device)
            torch.save(
                self.model.state_dict(),
                './model_weights/nostalgia_model_pretrained.pth'
            )

        self.theta_0 = {
            name: p.detach().clone() for name, p in self.model.backbone.named_parameters()
            if p.requires_grad
        }


    def EWC_regularization(
        self,
        lambda_ewc = None,
    ):
        lambda_ewc = lambda_ewc if lambda_ewc is not None else self.config.ewc_lambda
        if not self.ewc_fisher:
            return torch.tensor(0.0, device=self.config.device)

        loss = torch.tensor(0.0, device=self.config.device)

        for name, param in self.model.backbone.named_parameters():
            if name in self.ewc_fisher:
                loss += self.ewc_fisher[name] * (param - self.ewc_params[name]).pow(2).sum()

        return (lambda_ewc/2) * loss


    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)


    def prepare_all_datasets(self):
        tokenizer = self.model.backbone.tokenizer
        seq_collator = DataCollatorWithPadding(
            tokenizer,  # type: ignore
            padding=True
        )
        bs = self.config.batch_size

        # ---------- SST-2 ----------
        sst2 = load_dataset("glue", "sst2")
        sst2 = sst2.map(lambda x: tokenize_sst2(x, tokenizer))
        sst2.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        sst2_train = DataLoader(sst2["train"], batch_size=bs, shuffle=True, collate_fn=seq_collator) # type: ignore
        sst2_val = DataLoader(sst2["validation"], batch_size=bs, collate_fn=seq_collator)            # type: ignore

        # ---------- MNLI ----------
        mnli = load_dataset("glue", "mnli")
        mnli = mnli.map(lambda x: tokenize_mnli(x, tokenizer))
        mnli.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        mnli_train = DataLoader(mnli["train"], batch_size=bs, shuffle=True, collate_fn=seq_collator)     # type: ignore
        mnli_val = DataLoader(mnli["validation_matched"], batch_size=bs, collate_fn=seq_collator)        # type: ignore

        # ---------- CoNLL ----------
        conll = load_dataset("conll2003")
        label_names = conll["train"].features["ner_tags"].feature.names
        label2id = {i: i for i in range(len(label_names))}

        conll = conll.map(
            lambda x: tokenize_conll(x, tokenizer, label2id),
        )

        collator = DataCollatorForTokenClassification(tokenizer)  # type: ignore
        conll.set_format("torch")

        conll_train = DataLoader(
            conll["train"], batch_size=bs, shuffle=True, collate_fn=collator # type: ignore
        )
        conll_val = DataLoader(
            conll["validation"], batch_size=bs, collate_fn=collator         # type: ignore
        )

        # ---------- Register ----------
        self.datasets = {
            "SST2": (sst2_train, sst2_val),
            "MNLI": (mnli_train, mnli_val),
            "CoNLL": (conll_train, conll_val),
        }

        self.order_of_tasks = ["SST2", "MNLI", "CoNLL"]

        self.dataset_num_classes = {
            "SST2": 2,
            "MNLI": 3,
            "CoNLL": len(label_names),
        }

        self.task_types = {
            "SST2": "sequence",
            "MNLI": "sequence",
            "CoNLL": "token",
        }

        for task in self.datasets:
            self.model.add_task(
                task,
                self.dataset_num_classes[task],
                self.task_types[task],
            )

        self.epochs_per_task = {
            "SST2": 3,
            "MNLI": 3,
            "CoNLL": 5,
        }

    def retrain_task_head(
        self,
        task_name: str,
        epochs: int = 5,
    ):
        train_loader, val_loader = self.datasets[task_name]
        self.model.set_active_task(task_name)
        criterion = self.model.criterion

        # Freeze all parameters except task head
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        self.model.task_head_list[task_name].train()

        optimizer = torch.optim.Adam(
            self.model.task_head_list[task_name].parameters(),
            lr=self.config.learning_rate
        )

        for epoch in range(epochs):
            for input, target in train_loader:
                self.model.train()
                input, target = input.to(self.config.device), target.to(self.config.device)
                input = self.model.preprocess_inputs(input)

                optimizer.zero_grad()
                output = self.model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Validate after each epoch
            self.validate_dataset(val_loader, criterion, iteration=epoch, task_name=task_name)



    def validate_dataset(
            self,
            val_loader: DataLoader,
            criterion,
            iteration: int = 0,
            task_name: Optional[str] = None
    ):
        self.model.set_active_task(task_name)  # type: ignore

        if self.config.adapt_downstream_tasks and task_name not in self.finished_datasets:
            print(f"Retraining task head for {task_name} before validation.")
            assert task_name is not None, "Task name must be provided for retraining task head."
            self.retrain_task_head(task_name, epochs=5)

        total_loss = 0.0
        accuracy = 0.0
        for input, target in val_loader:
            self.model.eval()
            input, target = input.to(self.config.device), target.to(self.config.device)
            input = self.model.preprocess_inputs(input)
            with torch.no_grad():
                output = self.model(input)
                loss = criterion(output, target)
                total_loss += loss.item() * input.size(0)
                accuracy += (output.argmax(dim=1) == target).sum().item()

        loss = total_loss / len(val_loader.dataset) #type: ignore
        accuracy = accuracy / len(val_loader.dataset) #type: ignore

        self.writer.add_scalar( f'{task_name}/validation_loss', loss, iteration)
        self.writer.add_scalar( f'{task_name}/validation_accuracy', accuracy, iteration)


    def train_dataset(
        self,
        task_name,
        Q_prev=None,
        scaling=None,
        starting_step=0,
        epochs=1,
        log_deltas=False,
        mode="nostalgia",
    ):
        train_loader, _ = self.datasets[task_name]
        self.model.set_active_task(task_name)
        self.model.set_Q(Q_prev, scaling)

        criterion = self.model.criterion
        niter = starting_step

        optimizer = (
            self.model.configure_optimizers(self.writer, starting_step)
            if log_deltas else
            self.model.configure_optimizers()
        )

        for epoch in range(epochs):
            for batch in tqdm(train_loader, desc=f"{task_name}", ncols=80):
                self.model.train()

                batch = self.model.preprocess_inputs(batch)
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                optimizer.zero_grad()
                logits = self.model(**batch)

                if self.task_types[task_name] == "token":
                    loss = criterion(
                        logits.view(-1, logits.size(-1)),
                        batch["labels"].view(-1),
                    )
                else:
                    loss = criterion(logits, batch["labels"])

                if mode == "l2sp":
                    loss += self.model.l2sp_regularization(
                        self.theta_0, self.config.l2sp_lambda
                    )

                loss.backward()
                optimizer.step()

                self.writer.add_scalar(f"{task_name}/train_loss", loss.item(), niter)
                niter += 1

        return niter


    def train(self, erase_past=False):
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
                fisher_information = self.model.get_fisher_information(dataloader=train_loader)
                self.store_ewc_information(fisher_information)

            print(f"Immediately after training on {task_name}, total steps: {total_steps}")

            if self.config.mode == "nostalgia":
                # After training on the current task, compute the Hessian eigenspace
                Q_curr, Lambda_curr = compute_Q_for_task(
                    model=self.model,
                    train_loader=train_loader,
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
                    self.model._merge_and_unload_peft()
                    self.model._apply_peft()

            # Save model after each task
            self.save_model(f'./model_weights/nlp/nostalgia_model_after_{task_name}.pth')

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
        # log_dir=args.log_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        ewc_lambda=args.ewc_lambda,
        l2sp_lambda=args.l2sp_lambda,
        reset_lora=args.reset_lora,
        accumulate_mode=args.accumulate_mode,
        log_deltas=args.log_deltas,
        use_scaling=args.use_scaling,
        adapt_downstream_tasks=args.adapt_downstream_tasks,
    )

    config.log_dir = f'./logs/nostalgia_nlp_experiment/{config.mode}/{config.learning_rate}/{config.lora_r}/{config.hessian_eigenspace_dim}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'


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
    # experiment.model._merge_and_unload_peft()
    # experiment.model._apply_peft()



from datasets.imagenet import *
from .common import ImageFolderWithPaths
from typing import List, Optional
import torch
from torch.utils.data import Subset, RandomSampler, SequentialSampler

class ImageNetSplit(ImageNet):
    """
    ImageNet dataset for training and testing on only a subset of classes defined by a list of class indices.
    Supports distributed training and per-task projection for continual/class-incremental learning.
    """
    def __init__(
        self,
        location: str,
        preprocess,
        batch_size: int,
        num_workers: int = 4,
        distributed: bool = False,
        class_indices: Optional[List[int]] = None,
        classnames: str = 'openai',  # passed to base if needed
    ):
        if class_indices is None:
            raise ValueError("class_indices must be provided for ImageNetSplit")
        
        self.class_indices = sorted(class_indices)  # ensure deterministic order
        self.class_sublist = self.class_indices
        self.class_sublist_mask = [i in self.class_indices for i in range(1000)]
        
        # Pass to base class (which will call populate_train/test)
        super().__init__(
            preprocess=preprocess,
            location=location,
            batch_size=batch_size,
            num_workers=num_workers,
            classnames=classnames,
            distributed=distributed,
        )

    def get_train_sampler(self):
        # Filter indices to only images from the allowed classes
        if hasattr(self.train_dataset, 'targets'):
            indices = [i for i, target in enumerate(self.train_dataset.targets) if target in self.class_indices]
        else:
            # Fallback: extract from samples
            indices = [i for i, (_, target) in enumerate(self.train_dataset.samples) if target in self.class_indices]

        if not indices:
            raise ValueError(f"No training images found for classes {self.class_indices}")

        subset = Subset(self.train_dataset, indices)

        if self.distributed:
            return torch.utils.data.distributed.DistributedSampler(subset)
        else:
            return RandomSampler(subset)  # random order, no replacement

    def get_test_sampler(self):
        # Filter val indices to only images from the allowed classes
        if hasattr(self.test_dataset, 'targets'):
            indices = [i for i, target in enumerate(self.test_dataset.targets) if target in self.class_indices]
        else:
            indices = [i for i, (_, target) in enumerate(self.test_dataset.samples) if target in self.class_indices]

        if not indices:
            raise ValueError(f"No validation images found for classes {self.class_indices}")

        # Always use SequentialSampler for validation (deterministic, full subset)
        subset = Subset(self.test_dataset, indices)
        return SequentialSampler(subset)

    # Optional: helpers for projecting outputs/labels to the sub-task space
    def project_logits(self, logits, device):
        if logits.dim() == 2 and logits.size(1) == 1000:
            # Full 1000-class logits → slice to selected classes
            return logits[:, self.class_indices].to(device)
        return logits.to(device)  # already projected or different shape

    def project_labels(self, labels, device):
        # Map original ImageNet labels (0-999) → 0 to (num_classes-1)
        label_map = {orig: new for new, orig in enumerate(self.class_indices)}
        projected = [label_map[int(l)] for l in labels]
        return torch.tensor(projected, dtype=torch.long, device=device)

    # If you want to override classnames (for logging/display)
    @property
    def classnames(self):
        base_names = get_classnames('openai')  # or whatever your base uses
        return [base_names[i] for i in self.class_indices]







class ImageNetSplitTask1(ImageNetSplit):
    def __init__(self, location: str, preprocess, batch_size: int, num_workers: int = 4, distributed: bool = False):
        class_indices = list(range(0, 200))  # Classes 0-199
        super().__init__(location, preprocess, batch_size, num_workers, distributed, class_indices)


class ImageNetSplitTask2(ImageNetSplit):
    def __init__(self, location: str, preprocess, batch_size: int, num_workers: int = 4, distributed: bool = False):
        class_indices = list(range(200, 400))  # Classes 200-399
        super().__init__(location, preprocess, batch_size, num_workers, distributed, class_indices)


class ImageNetSplitTask3(ImageNetSplit):
    def __init__(self, location: str, preprocess, batch_size: int, num_workers: int = 4, distributed: bool = False):
        class_indices = list(range(400, 600))  # Classes 400-599
        super().__init__(location, preprocess, batch_size, num_workers, distributed, class_indices)

class ImageNetSplitTask4(ImageNetSplit):
    def __init__(self, location: str, preprocess, batch_size: int, num_workers: int = 4, distributed: bool = False):
        class_indices = list(range(600, 800))  # Classes 600-799
        super().__init__(location, preprocess, batch_size, num_workers, distributed, class_indices)


class ImageNetSplitTask5(ImageNetSplit):
    def __init__(self, location: str, preprocess, batch_size: int, num_workers: int = 4, distributed: bool = False):
        class_indices = list(range(800, 1000))  # Classes 800-999
        super().__init__(location, preprocess, batch_size, num_workers, distributed, class_indices)

def get_imagenet_splits(location: str, preprocess, batch_size: int, num_workers: int = 4, distributed: bool = False):
    return [
        ImageNetSplitTask1(location, preprocess, batch_size, num_workers, distributed),
        ImageNetSplitTask2(location, preprocess, batch_size, num_workers, distributed),
        ImageNetSplitTask3(location, preprocess, batch_size, num_workers, distributed),
        ImageNetSplitTask4(location, preprocess, batch_size, num_workers, distributed),
        ImageNetSplitTask5(location, preprocess, batch_size, num_workers, distributed),
    ]





from datasets.imagenet import *
from .common import ImageFolderWithPaths
from typing import List, Optional
import torch
from torch.utils.data import SubsetRandomSampler, RandomSampler, SequentialSampler




class ImageNetSplit(ImageNet):
    def __init__(
        self,
        split_labels: Optional[List[int]] = None,
        split_name: str = "custom_split",
        **kwargs,
    ):
        """
        split_indices: Optional list of dataset indices to include in this split.
                       If None, uses the full dataset.
        split_name: Name of the split (used for directory naming).
        """
        self.split_labels = split_labels 
        self.split_name = split_name
        super().__init__(**kwargs)

    def name(self) -> str:  # type: ignore
        return f"imagenet_{self.split_name}"

    def populate_train(self):
        traindir = os.path.join(self.location, "train")
        full_dataset = ImageFolderWithPaths(traindir, transform=self.preprocess)
        if self.split_labels is not None:
            split_indices = [
                idx for idx, (_, label) in enumerate(full_dataset.samples)
                if label in self.split_labels
            ]
            self.train_dataset = Subset(full_dataset, split_indices)
        else:
            self.train_dataset = full_dataset

        self.sampler = self.get_train_sampler()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=self.sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=(self.sampler is None),
        )

    def populate_test(self):
        testdir = os.path.join(self.location, "val")
        full_dataset = ImageFolderWithPaths(testdir, transform=self.preprocess)
        if self.split_labels is not None:
            split_indices = [
                idx for idx, (_, label) in enumerate(full_dataset.samples)
                if label in self.split_labels
            ]
            self.test_dataset = Subset(full_dataset, split_indices)
        else:
            self.test_dataset = full_dataset

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.get_test_sampler(),
        )



class ImageNetSplitTask1(ImageNetSplit):
    def __init__(self, **kwargs):
        split_labels = list(range(0, 200))
        super().__init__(split_labels=split_labels, split_name="task1", **kwargs)

class ImageNetSplitTask2(ImageNetSplit):
    def __init__(self, **kwargs):
        split_labels = list(range(200, 400))
        super().__init__(split_labels=split_labels, split_name="task2", **kwargs)

class ImageNetSplitTask3(ImageNetSplit):
    def __init__(self, **kwargs):
        split_labels = list(range(400, 600))
        super().__init__(split_labels=split_labels, split_name="task3", **kwargs)

class ImageNetSplitTask4(ImageNetSplit):
    def __init__(self, **kwargs):
        split_labels = list(range(600, 800))
        super().__init__(split_labels=split_labels, split_name="task4", **kwargs)

class ImageNetSplitTask5(ImageNetSplit):
    def __init__(self, **kwargs):
        split_labels = list(range(800, 1000))
        super().__init__(split_labels=split_labels, split_name="task5", **kwargs)


def get_imagenet_split_task_classes(
        preprocess,
        location,
        batch_size=64,
        num_workers=8,
        distributed=False
):
    kwargs = {
        "preprocess": preprocess,
        "location": location,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "distributed": distributed,
    }

    return [
        ImageNetSplitTask1(**kwargs),
        ImageNetSplitTask2(**kwargs),
        ImageNetSplitTask3(**kwargs),
        ImageNetSplitTask4(**kwargs),
        ImageNetSplitTask5(**kwargs),
    ]

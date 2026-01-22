import os
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Subset
import numpy as np

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames

class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai',
                 distributed=False):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.distributed = distributed

        self.populate_train()
        self.populate_test()
    
    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess,
            )
        sampler = self.get_train_sampler()
        self.sampler = sampler
        kwargs = {'shuffle' : True} if sampler is None else {}
        # print('kwargs is', kwargs)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_train_sampler(self):
        return torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.distributed else None


    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return 'imagenet'

class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass

def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)

class ImageNetSubsample(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)

class ImageNetSubsampleValClasses(ImageNet):
    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass
    
    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])
        
        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [self.class_sublist.index(int(label)) for label in labels]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


class ImageNet98p(ImageNet):

    def get_train_sampler(self):
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        #if os.path.exists(idx_file):
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)
        # else:
        #     idxs = np.zeros(len(self.train_dataset.targets))
        #     target_array = np.array(self.train_dataset.targets)
        #     for c in range(1000):
        #         m = target_array == c
        #         n = len(idxs[m])
        #         arr = np.zeros(n)
        #         arr[:26] = 1
        #         np.random.shuffle(arr)
        #         idxs[m] = arr
        #     with open(idx_file, 'wb') as f:
        #         np.save(f, idxs)

        idxs = (1 - idxs).astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])

        return sampler


class ImageNet2p(ImageNet):

    def get_train_sampler(self):
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)

        idxs = idxs.astype('int')
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler

class ImageNet2pShuffled(ImageNet):

    def get_train_sampler(self):
        print('shuffling val set.')
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
        return sampler


class ImageNetSubsampleTrainValClasses(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Call to set classnames based on sublist
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess,
        )
        sampler = self.get_train_sampler()
        self.sampler = sampler
        kwargs = {'shuffle': True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def get_train_sampler(self):
        if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'targets'):
            indices = [idx for idx, target in enumerate(self.train_dataset.targets) if target in self.class_sublist]
        else:
            # Fallback: Compute targets if not pre-set (assuming ImageFolderWithPaths sets samples)
            targets = [s[1] for s in self.train_dataset.samples]
            indices = [idx for idx, target in enumerate(targets) if target in self.class_sublist]

        if self.distributed:
            subset = Subset(self.train_dataset, indices)
            return torch.utils.data.distributed.DistributedSampler(subset)
        else:
            return SubsetRandomSampler(indices)

    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])
        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [self.class_sublist.index(int(label)) for label in labels]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)

class ImageNetClassIncremental(ImageNetSubsampleTrainValClasses):
    def __init__(self, start_class, end_class, *args, **kwargs):
        self.start_class = start_class
        self.end_class = end_class
        super().__init__(*args, **kwargs)

    def get_class_sublist_and_mask(self):
        class_sublist = list(range(self.start_class, self.end_class))
        class_sublist_mask = [i in class_sublist for i in range(1000)]
        return class_sublist, class_sublist_mask

# Task-specific classes
class ImageNetTask1(ImageNetClassIncremental):
    def __init__(self, *args, **kwargs):
        super().__init__(start_class=0, end_class=200, *args, **kwargs)

class ImageNetTask2(ImageNetClassIncremental):
    def __init__(self, *args, **kwargs):
        super().__init__(start_class=200, end_class=400, *args, **kwargs)

class ImageNetTask3(ImageNetClassIncremental):
    def __init__(self, *args, **kwargs):
        super().__init__(start_class=400, end_class=600, *args, **kwargs)

class ImageNetTask4(ImageNetClassIncremental):
    def __init__(self, *args, **kwargs):
        super().__init__(start_class=600, end_class=800, *args, **kwargs)

class ImageNetTask5(ImageNetClassIncremental):
    def __init__(self, *args, **kwargs):
        super().__init__(start_class=800, end_class=1000, *args, **kwargs)


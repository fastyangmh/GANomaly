# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.data_preparation import MyMNIST, MyCIFAR10, MyImageFolder, ImageLightningDataModule
from typing import Optional, Callable, Dict, Tuple, Any
import torch
import os


#def
def create_datamodule(project_parameters):
    if project_parameters.predefined_dataset:
        dataset_class = eval('My{}'.format(
            project_parameters.predefined_dataset))
    else:
        dataset_class = MyImageFolder
    return ImageLightningDataModule(
        root=project_parameters.root,
        predefined_dataset=project_parameters.predefined_dataset,
        classes=project_parameters.classes,
        max_samples=project_parameters.max_samples,
        batch_size=project_parameters.batch_size,
        num_workers=project_parameters.num_workers,
        device=project_parameters.device,
        transforms_config=project_parameters.transforms_config,
        target_transforms_config=project_parameters.target_transforms_config,
        dataset_class=dataset_class)


#class
class MyMNIST(MyMNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        normal_class = '0 - zero'
        self.class_to_idx = self.class_to_idx()
        # the index of normal is 0 and abnormal is 1
        if train:
            self.data = self.data[self.targets ==
                                  self.class_to_idx[normal_class]]
            self.targets = torch.zeros(len(self.data),
                                       dtype=self.targets.dtype)
        else:
            self.targets = torch.where(
                self.targets == self.class_to_idx[normal_class], 0, 1)
        # the index of normal is 0 and abnormal is 1
        self.class_to_idx = {v: 1 for v in self.class_to_idx.keys()}
        self.class_to_idx[normal_class] = 0
        self.classes = ['normal', 'abnormal']

    def class_to_idx(self) -> Dict[str, int]:
        return super().class_to_idx


class MyCIFAR10(MyCIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        normal_class = 'dog'
        self.targets = torch.tensor(self.targets)
        # the index of normal is 0 and abnormal is 1
        if train:
            self.data = self.data[self.targets ==
                                  self.class_to_idx[normal_class]]
            self.targets = torch.zeros(len(self.data),
                                       dtype=self.targets.dtype)
        else:
            self.targets = torch.where(
                self.targets == self.class_to_idx[normal_class], 0, 1)
        # the index of normal is 0 and abnormal is 1
        self.class_to_idx = {v: 1 for v in self.class_to_idx.keys()}
        self.class_to_idx[normal_class] = 0
        self.classes = ['normal', 'abnormal']


class MyImageFolder(MyImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)
        self.classes = ['normal', 'abnormal']
        self.class_to_idx = {k: idx for idx, k in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample, _ = super().__getitem__(index)
        path, _ = self.samples[index]
        relpath = os.path.relpath(path, self.root)
        label, filename = os.path.split(relpath)
        target = self.class_to_idx[label]
        return sample, target


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample and target
    print('the dimension of sample: {}'.format(x.shape))
    print(
        'the dimension of target: {}'.format(1 if type(y) == int else y.shape))

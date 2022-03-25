# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.predict import ImagePredictDataset
from src.model import create_model
import torch
from DeepLearningTemplate.data_preparation import parse_transforms
from typing import Any
from os.path import isfile
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


#class
class Predict:
    def __init__(self, project_parameters) -> None:
        self.model = create_model(project_parameters=project_parameters).eval()
        if project_parameters.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.device = project_parameters.device
        self.batch_size = project_parameters.batch_size
        self.num_workers = project_parameters.num_workers
        self.classes = project_parameters.classes
        self.loader = Image.open
        self.color_space = project_parameters.color_space

    def predict(self, filepath) -> Any:
        result = []
        fake_samples = []
        if isfile(path=filepath):
            # predict the file
            sample = self.loader(fp=filepath)
            # the transformed sample dimension is (1, in_chans, freq, time)
            sample = self.transform(sample)[None]
            if self.device == 'cuda' and torch.cuda.is_available():
                sample = sample.cuda()
            with torch.no_grad():
                score, sample_hat = self.model(sample)
                result.append([score.item()])
                fake_samples.append(sample_hat.cpu().data.numpy())
        else:
            # predict the file from folder
            dataset = ImagePredictDataset(root=filepath,
                                          loader=self.loader,
                                          transform=self.transform,
                                          color_space=self.color_space)
            pin_memory = True if self.device == 'cuda' and torch.cuda.is_available(
            ) else False
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=pin_memory)
            with torch.no_grad():
                for sample in tqdm(data_loader):
                    if self.device == 'cuda' and torch.cuda.is_available():
                        sample = sample.cuda()
                    score, sample_hat = self.model(sample)
                    result.append(score.tolist())
                    fake_samples.append(sample_hat.cpu().data.numpy())
        result = np.concatenate(result, 0).reshape(-1, 1)
        fake_samples = np.concatenate(fake_samples, 0)
        print(', '.join(self.classes))
        print(result)
        return result, fake_samples


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict file
    result = Predict(project_parameters=project_parameters).predict(
        filepath=project_parameters.root)

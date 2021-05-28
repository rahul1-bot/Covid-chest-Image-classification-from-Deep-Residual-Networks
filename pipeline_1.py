# -*- coding: utf-8 -*-
from __future__ import annotations
__authors__: list[str] = ['Rahul_Sawhney', 'leah_kumar_khan']
#$ exec(False) if not __pipeline1__.__dict__() or any('Rahul_Sawhney', 'Leah_kumar_khan',) not in __authors__ 

__doc__ =  r'''
    project Abstract: ...

    Project Control Flow: pipe1ine_1: Data Engineering
                                1) Class Dataset
                                2) Class DataAnalysis
                                3) Class DataPreprocess
                          
                                4) Class GPU_acceleration
                                5) class Image_Classification_Base 
                                6) class Hyperparams
                                7) Class CNNModel : @: general_CNN_model, UNet, ResNet, Inception
                                8) Class Train_test_fit
                          
                          pipeline_2: Machine Learning
                          pipeline_3: Evaluation Metrics
                                9) class Evaluate_Model
                                10) Class SaveModel
'''
#@ pipeline_1: Data Engineering
    #@: class API_call
    #@: class dataset
    #@: class Data Analysis 
    #       : batch_img_display
    #       : batch_img_rgb_display
    #       : histplot
    #       : ...
    #@: class DataPreprocess
    #       : RandomCrop 
    #       : RandomHorizontalFlip
    #       : RandomVerticalFlip
    #       : Normalize Imgs
    #       : Denormalize


# Python Imports
import typing
from typing import Any, NewType, Generator, Optional,  Union, ClassVar, Container
import os, warnings, sys
warnings.filterwarnings(action= 'ignore')


# Data Analysis Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as tFF


# DL Imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# torch typing scripts
_path =  NewType("_path", Any)
_transform = NewType("_transform", Any)
_img = NewType("_img", Any)
_criterion = NewType("_criterion", Any)
_optimizer = NewType("_optimizer", Any)
_loss = NewType("_loss", Any)
_layer = NewType("_layer", Any)
_activation = NewType("_activation", Any)
_text = NewType("_text", Any)
_plot = NewType("_plot", Any)
_loader = NewType("_loader", Any)
_recurse = NewType("_recurse", Any)

#@:  ------------------------------ Data Loader Iterator ---------------------------------------
class CovidDataset:
    def __init__(self, path: _path, sub_path: _path,  
                                    batch_size: Optional[int] = 14, 
                                    img_resolution: Optional[int] = 64, 
                                    transform: Optional[_transform] = None) -> None:
        self.path = path
        self.sub_path = sub_path
        self.batch_size = batch_size
        self.img_resolution = img_resolution
        if transform: 
            self.transform = transform
        self.categories: list[str] = ['Covid', 'Normal', 'Pneumonia']
        if sub_path == 'train':
            self.dataset: pd.DataFrame = self.get_data(path, 'train', self.categories)
        if sub_path == 'test':
            self.dataset: pd.DataFrame = self.get_data(path, 'test', self.categories)
        indexes: list[int] = [x for x in range(len(self.dataset))]
        self.index_batch: list[list[int]] = [
            indexes[i: i + batch_size] for i in range(0, len(indexes), batch_size)
        ]     


    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    __str__ = __repr__
    

    def __len__(self) -> int:
        return len(self.index_batch)
    

    def __getitem__(self, index: int) -> tuple[_img, str]:
        batch: list[int] = self.index_batch[index]
        size: tuple[int, ...] = (self.img_resolution, self.img_resolution)
        images: list[_img] = []
        labels: list[str] = []
        for i in batch:
            img: _img = Image.open(self.dataset.iloc[i].path).convert('LA').resize(size)
            img: np.ndarray = np.array(img)
            lbl: list[str] = [self.dataset.iloc[i].label]
            images.append(img)
            labels.append(np.array(lbl))
        images: torch.Tensor = torch.Tensor(images).type(torch.float32)
        images: torch.Tensor = images.permute(0, 3, 1, 2)
        labels: torch.Tensor = torch.tensor(labels).type(torch.float32)
        images: torch.Tensor = self.normalize(images)
        if self.transform:
           images: torch.Tensor = self.transform(images)
        return images, labels
    

    @classmethod
    def normalize(cls, img: torch.Tensor) -> torch.Tensor:
        return img / 255
    

    @classmethod
    def get_data(cls, path: _path, sub_path: _path, categories: list[str]) -> pd.DataFrame:
        covid: _path  = path + sub_path + "\\" + categories[0] + "\\"
        normal: _path = path + sub_path + "\\" + categories[1] + "\\"
        pneumonia: _path = path + sub_path + "\\" + categories[2] + "\\"
        
        covid_list: list[str] = [
            os.path.abspath(os.path.join(covid, p))
            for p in os.listdir(covid)
        ]
        normal_list: list[str] = [
            os.path.abspath(os.path.join(normal, p))
            for p in os.listdir(normal)
        ]
        pneumonia_list: list[str] = [
            os.path.abspath(os.path.join(pneumonia, p))
            for p in os.listdir(pneumonia)
        ]
        covid_labels: list[int] = [0 for _ in range(len(covid_list))]
        normal_labels: list[int] = [1 for _ in range(len(normal_list))]
        pneumonia_labels: list[int] = [2 for _ in range(len(pneumonia_list))]
        
        path: _path = covid_list + normal_list + pneumonia_list
        labels: list[int] = covid_labels + normal_labels + pneumonia_labels
        dataframe: pd.DataFrame = pd.DataFrame.from_dict({'path': path, 'label': labels})
        dataframe: pd.DataFrame = dataframe.sample(frac= 1)
        return dataframe



#@:  ---------------------------------- Data Analysis ------------------------------------------
class CovidAnalysis:
    labels_map: ClassVar[dict[int, str]] = {
        0: 'Covid',
        1: 'Normal',
        2: 'Pneumonia'
    }
    
    
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    __str__ = __repr__



    @classmethod
    def batch_imgs_display(cls, train_data_loader: _loader, batch_no: Optional[int] = 0) -> _plot:
        figure: _plot = plt.figure(figsize= (8, 8))
        cols, rows = 4, 4
        for i in range(1, cols * rows + 1):
            figure.add_subplot(rows, cols, i)
            label: int = int(train_data_loader[batch_no][1][i].numpy())
            plt.imshow(np.asarray(tFF.to_pil_image(train_data_loader[batch_no][0][i]).convert('RGB')))
            plt.title(cls.labels_map[label])
            plt.axis('off')
        plt.show()



    @classmethod
    def batch_imgs_rgb_display(cls, train_data_loader: _loader, batch_no: Optional[int] = 0) -> _plot:
        figure: _plot = plt.figure(figsize= (8, 8))
        cols, rows = 4, 4
        for i in range(1, cols * rows + 1):
            figure.add_subplot(rows, cols, i)
            label: int = int(train_data_loader[batch_no][1][i].numpy())
            plt.imshow(np.asarray(tFF.to_pil_image(train_data_loader[batch_no][0][i][0])))
            plt.title(cls.labels_map[label])
            plt.axis('off')
        plt.show()

    
    @classmethod
    def train_histplot(cls, training_loader: _loader) -> _plot:
        ...
    

    @classmethod
    def test_histplot(cls, test_loader: _loader) -> _plot:
        ...


#@:  ---------------------------- Class Data Preprocess -------------------------------------------
class CovidPreprocess:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', "Name", 'ObjectID'],
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    __str__ = __repr__


    @classmethod
    def train_transform_container(cls) -> Container[Module[_transform]]:
        transforms_container: Container[Module[_transform]] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(256, padding= 4, padding_mode= 'reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        return transforms_container

    
    @classmethod
    def test_transform_container(cls) -> Container[Module[_transform]]:
        transforms_container: Container[_transform] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        return transforms_container



#@: Driver code
if __name__.__contains__('__main__'):
    path: _path= "C:\\Users\\Lenovo\\OneDrive\\Desktop\\__Desktop\\covid\\Covid19-dataset\\" 
    sub_path: list[str] = ['train', 'test']
    training_data_loader: _loader = CovidDataset(path= path, 
                                                 sub_path= sub_path[0], 
                                                 batch_size= 20, 
                                                 transform= CovidPreprocess.train_transform_container())
    
    testing_data_loader: _loader = CovidDataset(path= path, 
                                                sub_path= sub_path[1], 
                                                batch_size= 20, 
                                                transform= CovidPreprocess.test_transform_container())
    
    # labels_map: dict[int, str] = {
    #     0: 'Covid',
    #     1: 'Normal',
    #     2: 'Pneumonia'
    # }
    # figure = plt.figure(figsize= (8, 8))
    # cols, rows = 4, 4
    # for i in range(1, cols * rows + 1):
    #     figure.add_subplot(rows, cols, i)
    #     label: int = int(training_data_loader[0][1][i].numpy())
    #     plt.imshow(np.asarray(F.to_pil_image(training_data_loader[0][0][i][0])))
    #     plt.title(labels_map[label])
    #     plt.axis('off')
    # plt.show()
    CovidAnalysis.batch_imgs_display(training_data_loader, 0)
    #all_labels: list[int] = [
     #       label for batch in training_data_loader for label in batch
     #   ]
    #print(all_labels)
    #
    #print(sys.path)
    
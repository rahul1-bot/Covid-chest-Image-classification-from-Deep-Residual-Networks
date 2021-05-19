# -*- coding: utf-8 -*-
# Last modified on: 19/5/21 

from __future__ import annotations
__authors__: list[str] = [ 'Rahul_Sawhney', 'Leah_Khan' ]

'''
project Abstract: ...

Project Control Flow: 1) Class Dataset
                      2) Class DataAnalysis
                      3) Class DataPreprocess
                      4) Class GPU_acceleration
                      5) class Image_Classification_Base
                      6) class Hyperparams
                      7) Class CNNModel : @: general_CNN_model, UNet, ResNet, Inception
                      8) Class Train_test_fit
                      9) class Evaluate_Model
                      10) Class SaveModel
'''
# Python Imports
import typing
__name__: typing.__name__ = '__main__'
from typing import Any, NewType, Generator, Optional, Union, ClassVar
import os, warnings
warnings.filterwarnings(action= 'ignore')


# Data Analysis Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


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


#############################################################
#@: Dataset Loader Iterator
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
        if sub_path == 'Training':
            self.dataset: pd.DataFrame = self.get_data(path, 'Training', self.categories)
        if sub_path == 'Testing':
            self.dataset: pd.DataFrame = self.get_data(path, 'Testing', self.categories)
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
    

    def __getitem__(self, index: int) -> dict[_img, str]:
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


#############################################################################################
#@: class DataAnalysis
class CovidAnalysis:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    __str__ = __repr__


    @classmethod
    def batch_imgs_display(cls, batch_number: Optional[int] = 1) -> _plot:
        ...
    

    @classmethod
    def batch_imgs_rgb_display(cls, batch_number: Optional[int]) -> _plot:
        ...
    
    
    @classmethod
    def histplot(cls) -> _plot:
        ...
    

    @classmethod
    def convolution_filtered(cls, ) -> _plot:
        ...


##################################################################################################
#@: Class DataPreprocess
class CovidPreprocess:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    __str__ = __repr__


    @classmethod
    def invertImage(cls, ) -> _transform:
        ...


    @classmethod
    def horizontal_flip(cls, ) -> _transform:
        ...


    @classmethod
    def vertical_flip(cls, ) -> _transform:
        ...


    @classmethod
    def random_crop(cls, ) -> _transform:
        ...


    @classmethod
    def denormalize_image(cls, ) -> _transform:
        ...


    @classmethod
    def image_augumentation(cls) -> _transform:
        ...


    @classmethod
    def image_segmentation(cls, ) -> _transform:
        ...

    
    @classmethod
    def image_burr(cls, ) -> transform:
        ...

    
    @classmethod
    def image_zoom_unzoom(cls, ) -> _transform:
        ...

    
    @classmethod
    def image_noice(cls, ) -> _transform:
        ...
        

##################################################################################
#@: Class GPU_Acceleration
class GPU_Acceleration:
    def __init__(self, train_loader: _loader, device: Any) -> None:
        self.train_loader = train_loader
        self.device = device
    

    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'],
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    __str__ = __repr__

    
    @classmethod
    def get_default_device(cls) -> str:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    

    @classmethod
    def to_device(cls, data: torch.Tensor, device: Any) -> _recurse:
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking= True)
    

    def __len__(self) -> int:
        return len(self.train_loader)
    

    def __iter__(self) -> Generator[_loader, None, None]:
        for data in self.train_loader:
            yield self.to_device(data, self.device)
    

    @classmethod
    def is_working_gpu(cls) -> bool:
        return True if cls.get_default_device() == 'cuda' else False
    

##################################################################################################
#@: Image Classifier Base 
class Image_Classifier_Base(nn.Module):
    def accuracy(self, outputs: Any, labels: str) -> float:
        _, preds = torch.max(outputs, dim= 1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    

    def training_step(self, batch: int) -> float:
        images, labels = batch
        out: Any = self(images)
        loss: _loss = F.cross_entropy(out, torch.max(labels, 1)[1])
        return loss

    
    def validation_step(self, batch: int) -> dict[str, float]:
        images, labels = batch
        out: Any = self(images)
        loss: _loss = F.cross_entropy(out, torch.max(labels, 1)[1])
        acc: float = self.accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    

    def validation_epoch_end(self, outputs: Any) -> dict[str, tuple[int|float, ...]]:
        batch_losses: list = [
            x['val_loss'] for x in outputs
        ]
        epoch_loss: Container = torch.stack(batch_losses).mean()
        batch_accs: list = [
            x['val_acc'] for x in outputs
        ]
        epoch_acc: float = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    
    def epoch_end(self, epoch: int, result: float) -> _text:
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))



###################################################################################################
#@: Class Model Hyperparams
class CNNHyperParams:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    __str__ = __repr__  
    
    #@ params
    epochs: int = 5
    optimizer: _optimizor = torch.optim.Adam
    learning_rate: float = 0.01
    criterion: _criterion = nn.CrossEntropyLoss()
    weight_decay: float = 0.01
    momentum: float = 0.9

    #@ Layers
    convolutional_1: _layer = nn.Conv2d(in_channels= 2, out_channels= 32, 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')
    
    convolutional_2: _layer = nn.Conv2d(in_channels= 32, out_channels= 64, 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')
    
    convolutional_3: _layer = nn.Conv2d(in_channels= 64, out_channels= 128, 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')
    
    convolutional_4: _layer = nn.Conv2d(in_channels= 128, out_channels= 32, 
                                        kernel_size= (3, 3), stride= 1, 
                                        padding= 0, dilation= 1, 
                                        groups= 1, bias= True, 
                                        padding_mode= 'zeros')

    
    pooling_1: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 2, 
                                     padding= 0, dilation= 1,
                                     return_indices= False,
                                     ceil_mode= False)
    
    pooling_2: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 2,
                                     padding= 0, dilation= 1,
                                     return_indices= False,
                                     ceil_mode= False)
    
    pooling_3: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 2,
                                     padding= 0, dilation= 1,
                                     return_indices= False,
                                     ceil_mode= False)

    pooling_4: _layer = nn.MaxPool2d(kernel_size= (2, 2), stride= 2, 
                                     padding= 0, dilation= 1,
                                     return_indices= False,
                                     ceil_mode= False)

                            
    linear_1: _layer = nn.Linear(in_features= 128, out_features= 64, bias= True)
    linear_2: _layer = nn.Linear(in_features= 64, out_features= 32, bias= True)
    linear_3: _layer = nn.Linear(in_features= 32, out_features= 16, bias= True)
    linear_4: _layer = nn.Linear(in_features= 16, out_features= 8, bias= True)
    linear_5: _layer = nn.Linear(in_features= 8, out_features= 4, bias= True)


    #@ Activation functions 
    relu: _activation = nn.ReLU()
    tanh: _activation = nn.Tanh()
    softmax: _activation = nn.Softmax()

    #@ neuron characteristics
    flatten = nn.Flatten()
    dropout = nn.Dropout(p= 0.3, inplace= False)

#################################################################################################
#@: Class Basic_CNN_Model
class CNNModel(Image_Classifier_Base):
    def __init__(self, num_classes: int) -> None:
        super(CNNModel, self).__init__()
        
        self.convolution_layers = nn.Sequential(
            CNNHyperParams.convolutional_1,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_1,

            CNNHyperParams.convolutional_2,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_2,

            CNNHyperParams.convolutional_3,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_3,

            CNNHyperParams.convolutional_4,
            CNNHyperParams.relu,
            CNNHyperParams.pooling_4
        )

        self.linear_layers = nn.Sequential(
            CNNHyperParams.linear_1,
            CNNHyperParams.tanh,

            CNNHyperParams.linear_2,
            CNNHyperParams.relu,

            CNNHyperParams.linear_3,
            CNNHyperParams.tanh,

            CNNHyperParams.linear_4,
            CNNHyperParams.relu,

            CNNHyperParams.linear_5
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.convolution_layers(x)
        x: torch.Tensor = CNNHyperParams.dropout(x)
        x: torch.Tensor = CNNHyperParams.flatten(x)
        x: torch.Tensor = self.linear_layers(x)
        return CNNHyperParams.softmax(x)



#@: Class ResNet
class _ResNet_Block(nn.Module):
    #super(ResNet_Block, self).__init__(*args, **kwargs)
    ...

class ResNet(Image_Classifier_Base, _ResNet_Block):
    #super(ResNet, self).__init__(*args, **kwargs)
    ...


#@: Class UNet
class _UNet_Block(nn.Module):
    #super(_UNet_Block, self).__init__(*args, **kwargs)
    ...

class UNet(Image_Classifier_Base, _UNet_Block):
    #super(UNet, self).__init__(*args, **kwargs)
    ...

#@: Class Inception
class _Inception_Block(nn.Module):
    #super(_Inception_Block, self).__init__(*args, **kwargs)
    ...

class Inception(Image_Classifier_Base, _Inception_Block):
    #super(Inception, self).__init__(*args, **kwargs)
    ...

###############################################################################################
#@: Class Train_test_fit
class Train_Test_fit:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    __str__ = __repr__


    @torch.no_grad()
    def evaluate(self, model: _model, val_loader: _loader) -> dict[str, float]: 
        model.eval()
        outputs: list[float] = [
            model.validation_step(batch) for batch in val_loader
        ]
        return model.validation_epoch_end(outputs)
        
    
    def get_learning_rate(self, optimizer: _optimizer) -> Any:
        for param_group in optimizer.param_groups:
            return param_group['lr']

    
    def fit_one_cycle(self, epochs: int, 
                            max_learning_rate: float, 
                            model: _model, 
                            train_loader: _loader, 
                            val_loader: _loader,  # test_loader
                            weight_decay: Optional[float|int] = 0,
                            grad_clip: Optional[float] = None, 
                            opt_function: Optional[_optimizer] = torch.optim.SGD) -> _text: 
        
        torch.cuda.empty_cache()
        history: list = []
        optimizer: _optimizer = opt_function(model.parameters(), max_learning_rate, weight_decay= weight_decay)
        #@ one-cycle LR scherudlar
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                 max_learning_rate, 
                                                 epochs= epochs, 
                                                 steps_per_epoch= len(train_loader))
        for epoch in range(epochs):
            model.train()
            train_losses: list = []
            lrs: list = []
            for batch in train_loader:
                loss: _loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
                optimizer.step()
                optimizer.zero_grad()

                lrs.append(self.get_learning_rate(optimizer))
                sched.step()

            #@ validation
            result: float = self.evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            model.epoch_end(epoch, result)
            history.append(result)
    
        return history

#########################################################################################
#@: Class Evaluate Models
class EvaluateCNN:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    __str__ = __repr__


    @classmethod
    def accuracy_vs_no_of_epochs(cls, history: dict[str, float]) -> _plot:
        accuracies: list[float] = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs No. of epochs')
        plt.show()
    

    @classmethod
    def loss_vs_no_of_epochs(cls, history: dict[str, float]) -> _plot:
        train_losses: list[float] = [
            x.get('train_loss') for x in history
        ]
        val_losses: list[float] = [
            x['val_loss'] for x in history
        ]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs No. of Epochs')
        plt.show()
    

    @classmethod
    def learning_rate_vs_batch_number(cls, history: dict[str, float]) -> _plot:
        lrs: np.ndarray = np.concatenate([
            x.get('lrs', []) for x in history
        ])
        plt.plot(lrs)
        plt.xlabel('Batch No.')
        plt.ylabel('Learnjing rate')
        plt.title('Learning Rate vs Batch No.')
        plt.show()

    
    @classmethod
    def model_report(cls) -> pd.DataFrame | _plot:
        ...


    @classmethod
    def confusion_matrix(cls) -> _plot:
        ...

###############################################################################################
#@: Class SaveModel
class SaveCNN:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })
    

    __str__ = __repr__

    @classmethod
    def save(cls, model: _model, model_name: Optional[str] = 'BrainMRI_Model.pt') -> None:
        torch.save(model.state_dict(), model_name)
    

    @classmethod
    def is_save_model(cls, model: _model) -> bool:
        ...
    

    @classmethod
    def load_model(cls, new_model: _model, old_model: str) -> _model:
        new_model.load_state_dict(torch.load(old_model))
        return new_model
    

    @classmethod
    def is_load_model(cls, new_model: _model, old_model: _model) -> bool:
        ...



#@: Driver code
if __name__.__contains__('__main__'):
    path: _path = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\covid\\Covid19-dataset"
    ...
    







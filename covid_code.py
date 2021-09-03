from __future__ import annotations
__author__: list[str] = ['Rahul Sawhney', 'Leah Khan']

__doc__: str = r'''
    Paper Title: ...
    Paper Abstract: ...
    Paper Published Limk: ...

'''
import warnings, os, copy, time
from tqdm import tqdm
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils




class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, path: 'dir_path', sub_path: str, 
                                         categories: list[str],
                                         transform: torchvision.transforms) -> None:
        self.path = path
        self.sub_path = sub_path
        self.categories = categories
        self.transform = transform
        self.dataset = self.get_data()



    def get_data(self) -> pd.DataFrame:
        covid_path = os.path.join(self.path, self.sub_path, self.categories[0])
        normal_path = os.path.join(self.path, self.sub_path, self.categories[1])
        pneumonia_path = os.path.join(self.path, self.sub_path, self.categories[2])

        covid_pathList = [
            os.path.abspath(os.path.join(covid_path, p)) for p in os.listdir(covid_path)
        ] 
        normal_pathList = [
            os.path.abspath(os.path.join(normal_path, p)) for p in os.listdir(normal_path)
        ]
        pneumonia_pathList = [
            os.path.abspath(os.path.join(pneumonia_path, p)) for p in os.listdir(pneumonia_path)
        ]

        covid_label = [0 for _ in range(len(covid_pathList))]
        normal_label = [1 for _ in range(len(normal_pathList))]
        pneumonia_label = [2 for _ in range(len(pneumonia_pathList))]

        all_imgPaths = covid_pathList + normal_pathList + pneumonia_pathList
        all_labels = covid_label + normal_label + pneumonia_label

        dataframe = pd.DataFrame.from_dict({'path': all_imgPaths, 'label': all_labels})
        dataframe = dataframe.sample(frac= 1)
        return dataframe
    


    def __len__(self) -> int:
        return len(self.dataset)
    


    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.transform is not None:
            image = Image.open(self.dataset.iloc[index].path).convert('RGB')
            image = self.transform(image)
            label = self.dataset.iloc[index].label
        return image, label




class CovidAnalysis:
    label_map: dict[str, int] = {
        0: 'Covid',
        1: 'Normal',
        2: 'Pneumonia'
    }


    def __init__(self, data: object, loader: object) -> None:
        self.data = data
        self.loader = loader



    def batchImg_display(self) -> 'plot':
        figure = plt.figure(figsize= (8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_index = torch.randint(len(self.data), size= (1,)).item()
            image, label = self.data[sample_index]
            figure.add_subplot(rows, cols, i)
            plt.title(self.label_map[int(label)])
            plt.imshow(np.asarray(transforms.ToPILImage()(image).convert('RGB')))
        plt.show()



class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         kernel_size: int = 3, 
                                         padding: int = 1,
                                         pool: bool = False) -> None:
        super(ConvBlock, self).__init__()
        layers: dict[str, object] = {
            'conv': nn.Conv2d(in_channels, out_channels, kernel_size, padding= padding, bias= False),
            'batch_norm': nn.BatchNorm2d(out_channels, eps= 1e-4),
            'relu': nn.ReLU(inplace= True)
        }
        if pool:
            layers['pool'] = nn.MaxPool2d(kernel_size= 4)
    
        self.block = nn.Sequential(*layers.values())



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)





class ResNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(ResNet, self).__init__()
        
        conv_layer_One: dict[str, object] = {
            'conv_1': ConvBlock(in_channels= in_channels, out_channels= 64),
            'conv_2': ConvBlock(in_channels= 64, out_channels= 128, pool= True)
        }
        conv_layer_Two: dict[str, object] = {
            'conv_3': ConvBlock(in_channels= 128, out_channels= 256, pool= True),
            'conv_4': ConvBlock(in_channels= 256, out_channels= 512, pool= True)
        }
        residual_One: dict[str, object] = {
            'res_1': ConvBlock(in_channels= 128, out_channels= 128),
            'res_2': ConvBlock(in_channels= 128, out_channels= 128)
        }
        residual_Two: dict[str, object] = {
            'res_3': ConvBlock(in_channels= 512, out_channels= 512),
            'res_4': ConvBlock(in_channels= 512, out_channels= 512)
        }
        final_layer: dict[str, object] = {
            'pool': nn.MaxPool2d(kernel_size= 4),
            'flat': nn.Flatten(),
            'dropout': nn.Dropout(p= 0.2),
            'dense': nn.Linear(in_features= 512, out_features= num_classes)

        }

        self.Conv_One = nn.Sequential(*conv_layer_One.values())
        self.Residual_One = nn.Sequential(*residual_One.values())
        self.Conv_Two = nn.Sequential(*conv_layer_Two.values())
        self.Residual_Two = nn.Sequential(*residual_Two.values())
        self.classifier = nn.Sequential(*final_layer.values())



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Conv_One(x)
        x = self.Residual_One(x) + x
        x = self.Conv_Two(x)
        x = self.Residual_Two(x) + x
        x = self.classifier(x)
        return x
    




class Model():
    def __init__(self, net: 'model', criterion: object, 
                                     optimizer: object, 
                                     num_epochs: int, 
                                     dataloaders: dict[str, object],
                                     dataset_sizes: dict[str, int], 
                                     device: torch.device) -> None:
        super(Model, self).__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
    


    def train_validate(self, history: bool = False) -> dict[str, float]| None:
        since = time.time()
        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_acc: float = 0.0
        self.history: dict[str, list] = {
            x: [] for x in ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        }

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()
                
                running_loss: float = 0.0
                running_corrects: int = 0

                for images, labels in self.dataloaders[phase]:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.net(images)
                        _, pred_labels = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(pred_labels == labels.data)

                epoch_loss: float = running_loss/ self.dataset_sizes[phase]
                epoch_acc: float =  running_corrects.double()/ self.dataset_sizes[phase]
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc)
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase ==  'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.net.state_dict())

            print()
        
        time_elapsed = time.time() - since
        print(f'Training Completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s') 
        print(f'Best Val Acc: {best_acc:.4f}')
        if history:
            return self.history


    

    def train_ValAcc(self) -> 'plot':
        train_acc_list = [float(x.cpu().numpy()) for x in self.history['train_acc']]
        test_acc_list = [float(x.cpu().numpy()) for x in self.history['val_acc']]
        plt.plot(train_acc_list, '-bx')
        plt.plot(test_acc_list, '-rx')
        plt.title('Model Accuracy Plot')
        plt.xlabel('No of Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'validation'], loc= 'best')
        plt.show()



    def train_valLoss(self) -> 'plot':
        train_loss_list = [float(x) for x in self.history['train_loss']]
        test_loss_list =  [float(x) for x in self.history['val_loss']]
        plt.plot(train_loss_list, '-bx')
        plt.plot(train_loss_list, '-bx')
        plt.plot(test_loss_list, '-rx')
        plt.title('Model Loss Plot')
        plt.xlabel('No of Epoch')
        plt.ylabel('Loss')
        plt.legend(['train', 'validation'], loc= 'best')
        plt.show()



    def confusion_matrix(self, class_names: list[str]) -> 'plot':
        n_classes: int = len(class_names)
        confusion_matrix = torch.zeros(n_classes, n_classes)
        with torch.no_grad():
            for images, labels in self.dataloaders['test']:
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred_labels = self.net(images)
                _, pred_labels = torch.max(pred_labels, 1)
                for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        
        plt.figure(figsize= (8, 5))
        df_cm = pd.DataFrame(confusion_matrix, index= class_names, columns= class_names).astype(int)
        df_cm = sns.heatmap(df_cm, annot= True, fmt= '.3g', cmap= 'Blues')
        df_cm.yaxis.set_ticklabels(df_cm.yaxis.get_ticklabels(), rotation= 0, ha= 'right', fontsize= 10)
        df_cm.xaxis.set_ticklabels(df_cm.xaxis.get_ticklabels(), rotation= 45, ha= 'right', fontsize= 10)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.show()






# driver code
if __name__.__contains__('__main__'):
    path: 'dir_path' = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\Covid19-dataset'
    sub_path: list[str] = ['train', 'test']
    categories: list[str] = ['Covid', 'Normal', 'Viral Pneumonia']

    transforms_list = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomRotation(360),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_data: object = CovidDataset(
                    path= path, 
                    sub_path= sub_path[0], 
                    categories= categories, 
                    transform= transforms_list
    )
    test_data: object = CovidDataset(
                    path= path, 
                    sub_path= sub_path[1], 
                    categories= categories, 
                    transform= transforms_list
    )
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= 10, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= 10)

    covid_plots = CovidAnalysis(data= train_data, loader= train_loader)
    covid_plots.batchImg_display()

    dataset_sizes: dict[str, int] = {
        'train': len(train_data),
        'test': len(test_data)
    }

    dataloaders: dict[str, object] = {
        'train': train_loader,
        'test': test_loader
    }

    device: torch.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ResNet(3, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)

    image_classification_model = Model(
                    net= model, 
                    criterion= criterion, 
                    optimizer= optimizer, 
                    num_epochs= 100, 
                    dataloaders= dataloaders, 
                    dataset_sizes= dataset_sizes, 
                    device= device
    )


    image_classification_model.train_validate()
    image_classification_model.train_ValAcc()
    image_classification_model.train_valLoss()
    image_classification_model.confusion_matrix(class_names= categories)





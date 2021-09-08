from __future__ import annotations
__author__: list[str] = ['Rahul Sawhney']

__doc__: str = r'''
    >>> Paper Title: 
            An Efficient Supervised Deep Learning
            Approach for Covid chest Image classification
            from Deep Residual Networks 
    
    >>> Paper Abstract: 
            With the outbreak of the COVID-19 and its
            various mutations making the infections faster and severe, it is
            becoming extremely important to determine the presence of
            COVID-19 infection in one’s body at a faster pace. Tests of
            Molecular, Antigen and Chest Scans are conducted to
            determine the presence of infection in the body, however, the
            molecular and antigen tests like RT-PCR require some time
            ranging from 1-5 days depending upon the availability of lab in
            the locality and how they run their tests with equipment. On the
            other hand, Chest scans like X-Ray and CT scans require lesser
            time of 10-15 minutes for detection by MDs. But due to the rise
            in cases and increase in demand for tests, radiologists and MDs
            find it harder to respond in time. Chest X-Rays are preferred
            for their less intensity and effective cost compared to CT scans.
            The model presented in this paper has operated on a total of 317
            images containing COVID-19, Viral Pneumonia and Normal
            Chest X-Ray images. The model achieves an accuracy of 99.5%
            in the testing phase for classification of a COVID-19 infected
            Chest X-Ray. The aim is to help reduce the time taken for
            identifying infected X-Rays thus helping conduct tests at a
            faster pace. 
    
'''
import warnings, os, copy, time
from tqdm import tqdm
warnings.filterwarnings('ignore')
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms, utils



#@: Custom Dataset Class
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





#@: Data Analysis Class
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





def conv3x3(in_planes: int, out_planes: int, stride: Optional[int] = 1) -> nn.Conv2d():
    return nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size= 3, 
        stride= stride,
        padding= 1, 
        bias= False
    )


def conv1x1(in_planes: int, out_planes: int, stride: Optional[int] = 1) -> nn.Conv2d():
    return nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size= 1, 
        stride= stride, 
        bias= False
    )




#@: ResNet Basic ConvBlock 
class BasicBlock(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(self, inplanes: int, planes: int, 
                                      stride: Optional[int] = 1, 
                                      downsample: Optional[bool] = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity: torch.Tensor = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out




#@: Custom build ResNet18 Model 
class ResNet(nn.Module):
    def __init__(self, block: object, layers: list[int], 
                                      num_classes: Optional[int] = 3, 
                                      zero_init_residual: Optional[bool] = False) -> None:
        super(ResNet, self).__init__()
        self.inplanes: int = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size= 7, stride= 2, padding= 3, bias= False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)




    def _make_layer(self, block: object, planes: int, blocks: int, stride: Optional[int] = 1) -> nn.Sequential():
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x





#@: Model Adaptor Class 
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




#@: Driver Code 
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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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

    model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)

    image_classification_model = Model(
                    net= model, 
                    criterion= criterion, 
                    optimizer= optimizer, 
                    num_epochs= 25, 
                    dataloaders= dataloaders, 
                    dataset_sizes= dataset_sizes, 
                    device= device
    )


    image_classification_model.train_validate()
    image_classification_model.train_ValAcc()
    image_classification_model.train_valLoss()
    image_classification_model.confusion_matrix(class_names= categories)





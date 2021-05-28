# -*- coding: utf-8 -*-
#@: Pipeline2: Machine Learning
    #@: class GPU_Acceleration
    #@: Class Image_Classifier_Base
    #@: Class Hyperparams
    #@: Class CNN_Model
    #@: Class ResNet
    #@: Class Inception
    #@: Class UNet
    #@: Class Train_Test_fit

from __future__ import annotations
from pipeline_1 import *
from typing import NewType, Any
import torchvision.models as models


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


#@: --------------------- GPU_Acceleration ---------------------------
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
            return [cls.to_device(x, device) for x in data]
        return data.to(device, non_blocking= True)
    

    def __len__(self) -> int:
        return len(self.train_loader)
    

    def __iter__(self) -> Generator[_loader, None, None]:
        for data in self.train_loader:
            yield self.to_device(data, self.device)
    

    @classmethod
    def is_working_gpu(cls) -> bool:
        return True if cls.get_default_device() == 'cuda' else False
    


def accuracy(outputs: Any, labels: str) -> float:
    _, preds = torch.max(outputs, dim= 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

#@: --------------- Image_Classifier_Base --------------------- 
class Image_Classifier_Base(nn.Module):
    def training_step(self, batch: int) -> float:
        images, labels = batch
        out: Any = self(images)
        #loss: _loss = F.cross_entropy(out, torch.max(labels, 1)[1])
        loss: _loss = F.cross_entropy(out, labels)
        return loss

    
    def validation_step(self, batch: int) -> dict[str, float]:
        images, labels = batch
        out: Any = self(images)
        #loss: _loss = F.cross_entropy(out, torch.max(labels, 1)[1])
        loss: _loss = F.cross_entropy(out, labels)
        acc: float = accuracy(out, labels)
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


#@:  ------------------ Model HyperParams --------------------- 
class CNNHyperParams:
    epochs: int = 5
    optimizer: _optimizor = torch.optim.Adam
    learning_rate: float = 0.01
    criterion: _criterion = nn.CrossEntropyLoss()
    weight_decay: float = 0.01
    momentum: float = 0.9


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

                            
    linear_1: _layer = nn.Linear(in_features= 6272, out_features= 64, bias= True)
    linear_2: _layer = nn.Linear(in_features= 64, out_features= 32, bias= True)
    linear_3: _layer = nn.Linear(in_features= 32, out_features= 16, bias= True)
    linear_4: _layer = nn.Linear(in_features= 16, out_features= 8, bias= True)
    linear_5: _layer = nn.Linear(in_features= 8, out_features= 4, bias= True)

 
    relu: _activation = nn.ReLU()
    tanh: _activation = nn.Tanh()
    softmax: _activation = nn.Softmax()

    flatten = nn.Flatten()
    dropout = nn.Dropout(p= 0.3, inplace= False)


#@:  ----------------------- Basic CNN Model -----------------------------------------

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
        #print(x.shape)
        x: torch.Tensor = self.convolution_layers(x)
        #print(x.shape)
        x: torch.Tensor = CNNHyperParams.dropout(x)
        #print(x.shape)
        x: torch.Tensor = CNNHyperParams.flatten(x)
        #print(x.shape)
        #x = x.view(-1, 32 * 2 * 2)
        x: torch.Tensor = self.linear_layers(x)
        #print(x.shape)
        #return F.softmax(x)
        #return CNNHyperParams.softmax(x)
        return x


#@:  ------ Pretrained Models ----- 
class PretrainedModels:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'],
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    __str__ = __repr__


    @classmethod
    def ResNet(cls) -> _model:
        model: _model = models.resnet18()
        return model
    

    @classmethod
    def Inception(cls) -> _model:
        model: _model = models.inception_v3()
        return model
    

#@: -------------------------------------------------
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: 
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(Image_Classifier_Base):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # input: 3 X 256 X 256
        self.conv1 = conv_block(in_channels, 64) # out: 64 X 256 X 256
        self.conv2 = conv_block(64, 128, pool=True) # out: 128 X 64 X 64
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True) # out: 256 X 16 X 16
        self.conv4 = conv_block(256, 512, pool=True) # out: 512 X 4 X 4
        self.res2 = nn.Sequential(conv_block(512, 512), 
                                  conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), # out: 512 X 1 X 1
                                        nn.Flatten(),  # out: 512
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))  # out: 3
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out



#@: ------------------------------ Train_test_Fit --------------------------------------------------
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
                            val_loader: _loader,  
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



#@: Driver Code
if __name__.__contains__('__main__'):
    ...
    

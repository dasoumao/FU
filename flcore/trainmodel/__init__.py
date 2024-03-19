import torchvision
from copy import deepcopy
from .models import *
from .bilstm import *
from .resnet import *
from .alexnet import *
from .mobilenet_v2 import *
from .transformer import *

def get_model(model, dataset, algorithm):
        if model == "cnn": # non-convex
            if "mnist" == dataset:
                model = LeNet()
            elif "fmnist" == dataset:
                model = LeNet()
            elif "cifar10" == dataset:
                model = torchvision.models.resnet18(pretrained=False, num_classes=10)
            elif "svhn" == dataset:
                model = torchvision.models.resnet18(pretrained=False, num_classes=10)
            elif "stl10" == dataset:
                model = torchvision.models.resnet18(pretrained=False, num_classes=10)
            elif "gtsrb" == dataset:
                model = mobilenet_v2(pretrained=False, num_classes=43)
            elif "tiny" == dataset: 
                model = torchvision.models.resnet50(pretrained=False, num_classes=200)
            elif "cifar100" == dataset:
                model = torchvision.models.resnet34(pretrained=False, num_classes=100)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        if algorithm == "FedAvg":
            head = deepcopy(model.fc)
            model.fc = torch.nn.Identity()
            model = BaseHeadSplit(model, head)
        else:
            raise NotImplementedError
        
        return model

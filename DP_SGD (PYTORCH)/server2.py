import flwr as fl
from pandas.core.arrays.integer import Int64Dtype
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.engine.training import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt 
from opacus import PrivacyEngine
from torch.autograd import Variable
from collections import OrderedDict


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

acc=[]
tot_loss=[]
num_workers = 0
batch_size = 64
# percentage of training set to use as validation
valid_size = 0.2
# transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
#                                       transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
#                                       transforms.RandomRotation(10),     #Rotates the image to a specified angel
#                                       transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
#                                       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
#                                       transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
#                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
#                                ])
 
 
transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
# training_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train) # Data augmentation is only done on training images
validation_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

# training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True) # Batch size of 100 i.e to work with 100 images at a time
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 64, shuffle=False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# num_train = len(validation_dataset)
# indices = list(range(num_train))
# np.random.shuffle(indices)
# split = int(np.floor(0.2 * num_train))
# train_idx, valid_idx = indices[split:], indices[:split]

# # define samplers for obtaining training and validation batches
# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)



def set_parameters(x,parameters):
        params_dict = zip(x.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        x.load_state_dict(state_dict, strict=True)

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # Use the last 5k training examples as a validation set
    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights):
        """Validate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        set_parameters(model,weights)  # Update model with the latest parameters
        with torch.no_grad():
            for data in validation_loader:
                # images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                images, labels = data
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                test_output = model(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
        print(accuracy)
        acc.append(accuracy)
        tot_loss.append(loss)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config

def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


model=Net()
def get_parameters(model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=2,
        min_eval_clients=3,
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(get_parameters(model)))
def start(strategy):
    fl.server.start_server("localhost:3000", config={"num_rounds": 3},strategy=strategy)
   
start(strategy)
print("Accuracy",acc)
print("Loss",tot_loss)

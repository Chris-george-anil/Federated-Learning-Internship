from collections import OrderedDict
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import sampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.autograd import Variable
import flwr as fl
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt 
from opacus import PrivacyEngine

transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])
 
 
transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

batch_size = 100
acc=[]
privacy=0

training_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train) # Data augmentation is only done on training images
validation_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

# training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True) # Batch size of 100 i.e to work with 100 images at a time
# validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle=False)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_train = len(training_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,shuffle=True,
                                           num_workers=0)
testloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,shuffle=True,
                                          num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def train(net,optimizer, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(DEVICE)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            # b_x = Variable(images)   # batch x
            # b_y = Variable(labels)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            # images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            images, labels = data
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            test_output = net(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    return loss, accuracy

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


class MNISTClient(fl.client.NumPyClient):
    def __init__(self,mod):
        self.net = mod
        self.criterion = torch.nn.CrossEntropyLoss()

        ## DP-SGD Implementation
        self.optimizer = torch.optim.SGD(mod.parameters(), lr=0.01, momentum=0.9)
        self.privacy_engine = PrivacyEngine(mod, sample_rate=batch_size/num_train,sample_size=num_train, noise_multiplier=2.0, max_grad_norm=1.0)
        self.privacy_engine.attach(self.optimizer)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net,self.optimizer, trainloader, epochs=1)
        global privacy
        privacy+=self.privacy_engine.get_privacy_spent(10**(-5))[0] # to calculate episilon value (delta taken as 10^(-5))
        return self.get_parameters(), len(trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, testloader)
        acc.append(accuracy)

        return float(loss), len(testloader), {"accuracy":float(accuracy)}

fl.client.start_numpy_client("localhost:3000", client=MNISTClient(Net()))
print("Final Acc",acc)
print("Privacy Loss",privacy)

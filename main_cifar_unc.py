'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
import random
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import os
import argparse
from scipy.stats import norm

from models import *
from utils import progress_bar

import numpy as np
from torch.utils.data import TensorDataset

class PCCLoss(nn.Module):
    def __init__(
        self, reduction='mean',
        pred_unc_eps = 1e-4, 
        resi_min = 1e-4, resi_max=1e3
    ) -> None:
        super(PCCLoss, self).__init__()
        self.reduction = reduction
        self.pred_unc_eps = pred_unc_eps
        self.resi_min = resi_min
        self.resi_max = resi_max
        self.std_min = 1e-5
        self.min_mean = 1e-5
        self.max_mean = 1 - 1e-5
        self.pred_unc_mean = None
        self.criterion = nn.CrossEntropyLoss(reduction='none')
		
    def forward(
        self, 
        pred, 
        pred_unc,  
        target
    ):

        pred = (pred).softmax(dim=1)
        resi = self.criterion(pred, target)
        resi = resi.clamp(min=self.resi_min, max=self.resi_max)
        with torch.no_grad():
            if self.pred_unc_mean == None:
                self.pred_unc_mean = pred_unc.mean(dim=0, keepdims = True)
            else:
                self.pred_unc_mean = 0.9*self.pred_unc_mean + pred_unc.mean(dim=0, keepdims = True)*0.1
        # # breakpoint
        cov = ((resi - resi.mean(dim=0, keepdims = True))*(pred_unc - pred_unc.mean(dim=0, keepdims = True)).squeeze(1)).mean()
        corr = cov/((resi.std()*pred_unc.std()).clamp(min=self.std_min))
        l = resi  + 1e-4 * (1 - corr) 
        # breakpoint()
        if self.reduction == 'mean':
            l = l.mean()
        elif self.reduction == 'sum':
            l = l.sum()
        else:
            print('Reduction not supported')
            return None
        if l != l:
            print('nan')
            breakpoint()
        return l, corr


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

training_size = 6000
test_size = 10000
num_epochs = 100

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with uncertainty')
parser.add_argument('-exp_name', default='cifar_unc', type=str, help='')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device:", device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print('==> Preparing data..')

# Load CIFAR-10.2
cifar102_path = './data/cifar_10.2/cifar102_test.npz'
cifar102 = np.load(cifar102_path)

images = cifar102['images']  # Shape: (10000, 32, 32, 3)
labels = cifar102['labels']  # Shape: (10000,)

# Convert to PyTorch tensors
images = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0  # Convert to [N, C, H, W] and normalize
labels = torch.tensor(labels).long()

# Apply the same normalization as original CIFAR-10
transform_test = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010))
images = torch.stack([transform_test(img) for img in images])

# Create DataLoader
cifar102_testset = TensorDataset(images, labels)
cifar102_testloader = torch.utils.data.DataLoader(
    cifar102_testset, batch_size=100, shuffle=False, num_workers=2)

# Load CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

random.seed(42)
training_size = min(training_size, len(trainset))
subset_indices = random.sample(range(len(trainset)), training_size)
trainset = Subset(trainset, subset_indices)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)




testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_size = min(test_size, len(testset))
subset_indices = random.sample(range(len(testset)), test_size)
testset = Subset(testset, subset_indices)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')

# net = ResNet18()
net = ResNet18_uncertainty()

# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# criterion = nn.CrossEntropyLoss()
criterion = PCCLoss(reduction='mean', pred_unc_eps=1e-4, resi_min=1e-4, resi_max=1e3)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch,loss_function):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    corr_ls = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, unc = net(inputs)
        loss, corr = loss_function(outputs, unc, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        corr_ls.append(corr.item())
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # breakpoint()
    print(np.mean(corr_ls))


def test_cifar_10(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, unc = net(inputs)
            loss, corr = criterion(outputs, unc, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = correct/total
    if acc > best_acc or epoch % 10 == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        checkpoint_path = 'checkpoint/cifar_unc_{timestamp}'.format(timestamp = timestamp)
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint_name = './checkpoint/cifar_unc_{timestamp}/ckpt_{epoch}.pth'
        torch.save(state, checkpoint_name.format(timestamp = timestamp, epoch = epoch))
        best_acc = acc
    return acc

def test_cifar_10_2(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(cifar102_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, unc = net(inputs)
            loss, corr = criterion(outputs, unc, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(cifar102_testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = correct/total
    return acc

def safe_probit(p):
    epsilon = 1e-6
    p = min(max(p, epsilon), 1 - epsilon)
    return norm.ppf(p)

log_path = 'log/cifar_unc_{timestamp}'.format(timestamp = timestamp)
if not os.path.isdir(log_path):
    os.mkdir(log_path)

# main
cifar10_accuracies = []
cifar102_accuracies = []
cifar10_probits = []
cifar102_probits = []
# Train and test the model
for epoch in range(start_epoch, start_epoch + num_epochs):
    train(epoch,criterion)
    test_acc = test_cifar_10(epoch)
    ood_test_acc = test_cifar_10_2(epoch)
    cifar10_accuracies.append(test_acc * 100)
    cifar102_accuracies.append(ood_test_acc * 100)
    cifar10_probits = [safe_probit(p/100) for p in cifar10_accuracies]
    cifar102_probits = [safe_probit(p/100) for p in cifar102_accuracies]
    scheduler.step()

# plot the accuracies
plt.figure(figsize=(8, 6))
plt.plot(cifar10_accuracies, cifar102_accuracies, marker='o')
plt.xlabel('CIFAR-10 Accuracy (%)')
plt.ylabel('CIFAR-10.2 Accuracy (%)')
plt.title('CIFAR-10 vs CIFAR-10.2 Accuracy per Epoch')
plt.grid(True)
plt.tight_layout()
plot_path = 'log/cifar_unc_{timestamp}/acc_relationship.png'
plt.savefig(plot_path.format(timestamp = timestamp))  # Save to file
plt.show()

# plot the probits  
plt.figure(figsize=(8, 6))
plt.plot(cifar10_probits, cifar102_probits, marker='o')
plt.xlabel('CIFAR-10 Probit')
plt.ylabel('CIFAR-10.2 Probit')
plt.title('CIFAR-10 vs CIFAR-10.2 Probit per Epoch')
plt.grid(True)
plt.tight_layout()
plot_path = 'log/cifar_unc_{timestamp}/probit_relationship.png'
plt.savefig(plot_path.format(timestamp = timestamp))  # Save to file
plt.show()

# Save accuracies to CSV
df = pd.DataFrame({
    'epoch': list(range(start_epoch, start_epoch + num_epochs)),
    'cifar10_acc': cifar10_accuracies,
    'cifar102_acc': cifar102_accuracies
})
accuracies_path = 'log/cifar_unc_{timestamp}/accuracies.csv'
df.to_csv(accuracies_path.format(timestamp = timestamp), index=False)
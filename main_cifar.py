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
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

training_size = 60000
test_size = 10000
num_epochs = 100

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-exp_name', default='cifar', type=str, help='')

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
labels = cifar102['labels']  # Shape: (2000,)

# Convert to PyTorch tensors
images = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0  # Convert to [N, C, H, W] and normalize
labels = torch.tensor(labels).long()

# Apply the same normalization as original CIFAR-10
transform_test = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010))
images = torch.stack([transform_test(img) for img in images])

# Optional: select a subset
# subset_indices = random.sample(range(len(images)), 1000)
# images = images[subset_indices]
# labels = labels[subset_indices]

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
test_size = min(test_size, len(trainset))
subset_indices = random.sample(range(len(testset)), test_size)
testset = Subset(testset, subset_indices)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def compute_ece(probs, labels, n_bins=15):
    confidences, predictions = probs.max(1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(probs.device)
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        prop_in_bin = mask.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[mask].float().mean()
            avg_confidence_in_bin = confidences[mask].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

# Training
def train(epoch, loss_function):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_targets = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        # Store outputs for ECE
        probs = torch.softmax(outputs.detach(), dim=1)
        all_probs.append(probs)
        all_targets.append(targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                         train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Concatenate all for ECE
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    ece = compute_ece(all_probs, all_targets)

    print(f"Train ECE: {ece:.4f}")


def test_cifar_10(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            # Store outputs for ECE
            probs = torch.softmax(outputs.detach(), dim=1)
            all_probs.append(probs)
            all_targets.append(targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    avg_loss = test_loss / total
    acc = correct/total
    if acc > best_acc or epoch % 10 == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        checkpoint_path = 'checkpoint/cifar_{timestamp}'.format(timestamp = timestamp)
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint_name = './checkpoint/cifar_{timestamp}/ckpt_{epoch}.pth'
        torch.save(state, checkpoint_name.format(timestamp = timestamp, epoch = epoch))
        best_acc = acc
    # Concatenate all for ECE
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    ece = compute_ece(all_probs, all_targets)
    return acc, avg_loss, ece

def test_cifar_10_2(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_probs = []
    all_targets = []
    # Test on CIFAR-10.2
    print('\nTesting on CIFAR-10.2...')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(cifar102_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            # Store outputs for ECE
            probs = torch.softmax(outputs.detach(), dim=1)
            all_probs.append(probs)
            all_targets.append(targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(cifar102_testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    avg_loss = test_loss / total
    acc = correct/total
    # Concatenate all for ECE
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    ece = compute_ece(all_probs, all_targets)
    return acc, avg_loss, ece

def safe_probit(p):
    epsilon = 1e-6
    p = min(max(p, epsilon), 1 - epsilon)
    return norm.ppf(p)

log_path = 'log/cifar_{timestamp}'.format(timestamp = timestamp)
if not os.path.isdir(log_path):
    os.mkdir(log_path)

# main
cifar10_accuracies = []
cifar102_accuracies = []
cifar10_probits = []
cifar102_probits = []
cifar10_losses = []
cifar102_losses = []
cifar10_ece = []
cifar102_ece = []
# Train and test the model
for epoch in range(start_epoch, start_epoch + num_epochs):
    train(epoch,criterion)
    test_acc, test_loss, test_ece = test_cifar_10(epoch)
    ood_test_acc, ood_test_loss, ood_test_ece = test_cifar_10_2(epoch)
    cifar10_accuracies.append(test_acc * 100)
    cifar102_accuracies.append(ood_test_acc * 100)
    cifar10_probits = [safe_probit(p/100) for p in cifar10_accuracies]
    cifar102_probits = [safe_probit(p/100) for p in cifar102_accuracies]
    cifar10_losses.append(test_loss)
    cifar102_losses.append(ood_test_loss)
    cifar102_ece.append(ood_test_ece)
    cifar10_ece.append(test_ece)
    scheduler.step()

# plot the accuracies
plt.figure(figsize=(8, 6))
plt.plot(cifar10_accuracies, cifar102_accuracies, marker='o', linestyle='None')
plt.xlabel('CIFAR-10 Accuracy (%)')
plt.ylabel('CIFAR-10.2 Accuracy (%)')
plt.title('CIFAR-10 vs CIFAR-10.2 Accuracy per Epoch')
plt.grid(True)
plt.tight_layout()
plot_path = 'log/cifar_{timestamp}/acc_relationship.png'
plt.savefig(plot_path.format(timestamp = timestamp))  # Save to file
plt.show()

# plot the probits 
plt.figure(figsize=(8, 6))
plt.plot(cifar10_probits, cifar102_probits, marker='o', linestyle='None')
plt.xlabel('CIFAR-10 Probit')
plt.ylabel('CIFAR-10.2 Probit')
plt.title('CIFAR-10 vs CIFAR-10.2 Probit per Epoch')
plt.grid(True)
plt.tight_layout()
plot_path = 'log/cifar_{timestamp}/probit_relationship.png'
plt.savefig(plot_path.format(timestamp = timestamp))  # Save to file
plt.show()

# plot the losses
plt.figure(figsize=(8, 6))
plt.plot(cifar10_losses, cifar102_losses, marker='o', linestyle='None')
plt.xlabel('CIFAR-10 Loss')
plt.ylabel('CIFAR-10.2 Loss')
plt.title('CIFAR-10 vs CIFAR-10.2 Loss per Epoch')
plt.grid(True)
plt.tight_layout()
plot_path = 'log/cifar_{timestamp}/loss_relationship.png'
plt.savefig(plot_path.format(timestamp = timestamp))  # Save to file
plt.show()

# plot the ECE
plt.figure(figsize=(8, 6))
plt.plot(range(start_epoch, start_epoch + num_epochs), cifar10_ece, label='CIFAR-10 ECE', marker='o')
plt.plot(range(start_epoch, start_epoch + num_epochs), cifar102_ece, label='CIFAR-10.2 ECE', marker='o')
plt.xlabel('Epoch')
plt.ylabel('ECE')
plt.title('Expected Calibration Error (ECE) per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
ece_plot_path = 'log/cifar_{timestamp}/ece_plot.png'
plt.savefig(ece_plot_path.format(timestamp = timestamp))  # Save to file
plt.show()

# plot correlation between ECE of two datasets
plt.figure(figsize=(8, 6))
plt.plot(cifar10_ece, cifar102_ece, marker='o', linestyle='None')
plt.xlabel('CIFAR-10 ECE')
plt.ylabel('CIFAR-10.2 ECE')
plt.title('CIFAR-10 vs CIFAR-10.2 ECE')
plt.grid(True)
plt.tight_layout()
ece_corr_path = 'log/cifar_{timestamp}/ece_correlation.png'
plt.savefig(ece_corr_path.format(timestamp = timestamp))  # Save to file
plt.show()

# Save accuracies to CSV
df = pd.DataFrame({
    'epoch': list(range(start_epoch, start_epoch + num_epochs)),
    'cifar10_acc': cifar10_accuracies,
    'cifar102_acc': cifar102_accuracies,
    'cifar10_probit': cifar10_probits,
    'cifar102_probit': cifar102_probits,
    'cifar10_loss': cifar10_losses,
    'cifar102_loss': cifar102_losses,
    'cifar10_ece': cifar10_ece,
    'cifar102_ece': cifar102_ece
})
accuracies_path = 'log/cifar_{timestamp}/accuracies.csv'
df.to_csv(accuracies_path.format(timestamp = timestamp), index=False)
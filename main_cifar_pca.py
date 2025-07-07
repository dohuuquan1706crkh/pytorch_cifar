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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from torch.utils.data import TensorDataset
import os
import argparse
from scipy.stats import norm
from models import *
from utils import *

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
training_size = 6000
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
cifar102_train_path = './data/cifar_10.2/cifar102_train.npz'
cifar102_train = np.load(cifar102_train_path)
cifar102_path = './data/cifar_10.2/cifar102_test.npz'
cifar102 = np.load(cifar102_path)
images_train = cifar102_train['images']  # Shape: (10000, 32, 32, 3)
labels_train = cifar102_train['labels']  # Shape: (10000,)
images = cifar102['images']  # Shape: (10000, 32, 32, 3)
labels = cifar102['labels']  # Shape: (10000,)
# Convert to PyTorch tensors
images_train = torch.tensor(images_train).permute(0, 3, 1, 2).float() / 255.0  # Convert to [N, C, H, W] and normalize
labels_train = torch.tensor(labels_train).long()
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
cifar102_trainset = TensorDataset(images_train, labels_train)
training_size = min(training_size, len(cifar102_trainset))
subset_indices = random.sample(range(len(cifar102_trainset)), training_size)
cifar102_trainset = Subset(cifar102_trainset, subset_indices)
# labels_102 = labels[subset_indices]
cifar102_trainloader = torch.utils.data.DataLoader(
    cifar102_trainset, batch_size=100, shuffle=True, num_workers=2)
# Load CIFAR-10.2 test set
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
# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('/raid/quandh/pytorch-cifar/checkpoint/cifar/ckpt_99.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
outputs_10 = extract_latent_representations(trainloader, net, device)
outputs_102 = extract_latent_representations(cifar102_trainloader, net, device)

print(outputs_10.shape, outputs_102.shape)

criterion = nn.CrossEntropyLoss()
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
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_cifar_10():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = correct/total
    return acc

def test_cifar_10_2():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(cifar102_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(cifar102_testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = correct/total
    return acc

def test_cifar_10_pca(transform_matrix, bias):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            latent = net.module.latent(inputs)
            outputs = net.module.output((latent-bias) @ transform_matrix.T + bias)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = correct/total
    return acc

def test_cifar_10_2_pca(transform_matrix, bias):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(cifar102_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            latent = net.module.latent(inputs)
            outputs = net.module.output((latent-bias) @ transform_matrix.T + bias)
            loss = criterion(outputs, targets)
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

log_path = 'log/cifar_{timestamp}'.format(timestamp = timestamp)
if not os.path.isdir(log_path):
    os.mkdir(log_path)
# Extract latent representations


# Combine outputs from both datasets
outputs_combined = np.concatenate([outputs_10, outputs_102], axis=0)
domain_labels = np.array([0] * len(outputs_10) + [1] * len(outputs_102))
labels_combined = np.concatenate([trainset.dataset.targets[:len(outputs_10)],
                                  cifar102_trainset[:len(outputs_102)][1].numpy()], axis=0)
# Perform PCA
pca = PCA(n_components=512)
pca_results = pca.fit_transform(outputs_combined)
pca_modified = pca_results.copy()
transformation_matrix = pca.components_  # shape: (n_components, total components)

print("Transformation matrix shape:", transformation_matrix.shape)
results = []
for i in range(transformation_matrix.shape[0]):
    transformation_matrix[:, 512-i-1:] = 0
    # print(transformation_matrix)

    transform_matrix = transformation_matrix.T @ transformation_matrix
    # print(transform_matrix.shape)
    transform_matrix = torch.tensor(transform_matrix).float().to(device)
    bias = pca.mean_
    bias = torch.tensor(bias).float().to(device)


    # # test_acc = test_cifar_10()
    # # ood_test_acc = test_cifar_10_2()
    # # print(f'Test accuracy on CIFAR-10: {test_acc * 100:.2f}%')
    # # print(f'Test accuracy on CIFAR-10.2: {ood_test_acc * 100:.2f}%')
    test_acc_pca = test_cifar_10_pca(transform_matrix, bias)
    ood_test_acc_pca = test_cifar_10_2_pca(transform_matrix, bias)
    print("results when eliminating {} components:".format(i+1))
    print(f'Test accuracy on CIFAR-10 with PCA: {test_acc_pca * 100:.2f}%')
    print(f'Test accuracy on CIFAR-10.2 with PCA: {ood_test_acc_pca * 100:.2f}%')
    results.append({
        'num_components': i + 1,
        'cifar10_acc': test_acc_pca * 100,
        'cifar102_acc': ood_test_acc_pca * 100
    })
# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(log_path, f'results_{args.exp_name}_2.csv'), index=False)
# Plot probit relationship
plot_probit_relationship([results_df], [args.exp_name])
# plot relationship between accuracy and number of components
plt.figure(figsize=(8, 6))
plt.plot(results_df['num_components'], results_df['cifar10_acc'], marker='o', label='CIFAR-10')
plt.plot(results_df['num_components'], results_df['cifar102_acc'], marker='o', label='CIFAR-10.2')
plt.xlabel('Number of PCA Components Eliminated')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Number of PCA Components Eliminated')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(log_path, f'accuracy_vs_components_{args.exp_name}_2.png'))
plt.show()
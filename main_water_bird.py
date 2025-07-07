'''Train water_bird with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, Dataset, DataLoader
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
from PIL import Image

random.seed(42)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

training_size = 10000
test_size = 10000
num_epochs = 10

parser = argparse.ArgumentParser(description='PyTorch waterbird Training')
parser.add_argument('-exp_name', default='water_bird', type=str, help='')

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device:", device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print('==> Preparing data..')

# Read the CSV file into a DataFrame
file_path = "./waterbird_complete95_forest2water2/metadata.csv"  # Replace with your file path
df = pd.read_csv(file_path)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize((512, 512)), 
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)), 
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Define a custom dataset class
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Assuming 'img_filename' and 'place_filename' contain image paths
        img_path = self.dataframe.iloc[idx]['img_filename']
        # place_path = self.dataframe.iloc[idx]['place_filename']
        label = self.dataframe.iloc[idx]['y']  # Your target variable
        img_path = os.path.join('./waterbird_complete95_forest2water2/', img_path)
        # place_path = os.path.join('./waterbird_complete95_forest2water2/', place_path)
        # breakpoint()
        
        # Load images (replace with actual image loading)
        image = Image.open(img_path)
        # place_image = Image.open(place_path)
        
        # Placeholder for image data: replace this with your actual image loading logic
        # image = img_path
        # place_image = place_path

        if self.transform:
            image = self.transform(image)
            # place_image = self.transform(place_image)
        # Define transform
        # to_tensor = transforms.ToTensor()
        # Convert to tensor
        # image = to_tensor(image)
        # return image, place_image, label
        return image, label


# Split the data based on 'split' and 'place'
train_df = df[(df['split'] == 0) | (df['split'] == 1)]
# valid_df = df[(df['split'] == 1)]
test_df = df[(df['split'] == 2)]

train_df_place_0 = train_df[train_df['place'] == 0]
train_df_place_1 = train_df[train_df['place'] == 1]

# valid_df_place_0 = valid_df[valid_df['place'] == 0]
# valid_df_place_1 = valid_df[valid_df['place'] == 1]

test_df_place_0 = test_df[test_df['place'] == 0]
test_df_place_1 = test_df[test_df['place'] == 1]
# breakpoint()


# Create datasets
# Example: replace with your actual transforms
# transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

train_dataset_0 = ImageDataset(train_df_place_0, transform=transform_train)
train_dataset_1 = ImageDataset(train_df_place_1, transform=transform_train)

# valid_dataset_0 = ImageDataset(valid_df_place_0, transform=transform_test)
# valid_dataset_1 = ImageDataset(valid_df_place_1, transform=transform_test)

test_dataset_0 = ImageDataset(test_df_place_0, transform=transform_test)
test_dataset_1 = ImageDataset(test_df_place_1, transform=transform_test)


# Create data loaders
batch_size = 32 # Example batch size
random.seed(42)
training_size = min(training_size, len(train_dataset_0))
subset_indices = random.sample(range(len(train_dataset_0)), training_size)
train_dataset_0 = Subset(train_dataset_0, subset_indices)

test_size = min(test_size, len(test_dataset_0))
subset_indices = random.sample(range(len(test_dataset_0)), test_size)
test_dataset_0 = Subset(test_dataset_0, subset_indices)

trainloader = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)
trainloader_ood = DataLoader(train_dataset_0, batch_size=batch_size, shuffle=True)

# validloader = DataLoader(valid_dataset_1, batch_size=batch_size, shuffle=False)
# validloader_ood = DataLoader(valid_dataset_0, batch_size=batch_size, shuffle=False)

testloader = DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False)
testloader_ood = DataLoader(test_dataset_0, batch_size=batch_size, shuffle=False)

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
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
net = SimpleDLA()
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
# weight = torch.tensor([1.0, 4.0]).to(device)
criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.4, 1.0]).to(device))
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



# Training
def train_water_bird(epoch,loss_function):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # breakpoint()
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_water_bird(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # breakpoint()
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
    if acc > best_acc or epoch % 10 == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        checkpoint_path = 'checkpoint/water_bird_{timestamp}'.format(timestamp = timestamp)
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint_name = './checkpoint/water_bird_{timestamp}/ckpt_{epoch}.pth'
        torch.save(state, checkpoint_name.format(timestamp = timestamp, epoch = epoch))
        best_acc = acc
    return acc

def test_water_bird_ood(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader_ood):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader_ood), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = correct/total
    return acc

def safe_probit(p):
    epsilon = 1e-6
    p = min(max(p, epsilon), 1 - epsilon)
    return norm.ppf(p)

log_path = 'log/water_bird_{timestamp}'.format(timestamp = timestamp)
if not os.path.isdir(log_path):
    os.mkdir(log_path)

# main
water_bird_accuracies = []
water_bird_ood_accuracies = []
water_bird_probits = []
water_bird_ood_probits = []
# Train and test the model
for epoch in range(start_epoch, start_epoch + num_epochs):
    train_water_bird(epoch,criterion)
    test_acc = test_water_bird(epoch)
    ood_test_acc = test_water_bird_ood(epoch)
    water_bird_ood_accuracies.append(ood_test_acc * 100)
    water_bird_accuracies.append(test_acc * 100)
    water_bird_probits = [safe_probit(p/100) for p in water_bird_accuracies]
    water_bird_ood_probits = [safe_probit(p/100) for p in water_bird_ood_accuracies]
    scheduler.step()

# plot the accuracies
plt.figure(figsize=(8, 6))
plt.plot(water_bird_accuracies, water_bird_ood_accuracies, marker='o')
plt.xlabel('water_bird_accuracies Accuracy (%)')
plt.ylabel('water_bird_ood_accuracies Accuracy (%)')
plt.title('water_bird Accuracy per Epoch')
plt.grid(True)
plt.tight_layout()
plot_path = 'log/water_bird_{timestamp}/acc_relationship.png'
plt.savefig(plot_path.format(timestamp = timestamp))  # Save to file
plt.show()

# plot the probits  
plt.figure(figsize=(8, 6))
plt.plot(water_bird_probits, water_bird_ood_probits, marker='o')
plt.xlabel('water_bird Probit')
plt.ylabel('water_bird_ood Probit')
plt.title('water_bird Probit per Epoch')
plt.grid(True)
plt.tight_layout()
plot_path = 'log/water_bird_{timestamp}/probit_relationship.png'
plt.savefig(plot_path.format(timestamp = timestamp))  # Save to file
plt.show()

# Save accuracies to CSV
df = pd.DataFrame({
    'epoch': list(range(start_epoch, start_epoch + num_epochs)),
    'water_bird_acc': water_bird_accuracies,
    'water_bird_ood_acc': water_bird_ood_accuracies
})
accuracies_path = 'log/water_bird_{timestamp}/accuracies.csv'
df.to_csv(accuracies_path.format(timestamp = timestamp), index=False)
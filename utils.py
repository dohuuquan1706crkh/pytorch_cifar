'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
from scipy.stats import norm

import torch.nn as nn
import torch.nn.init as init
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def plot_tsne(outputs, labels, title, path):
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=30, perplexity=50)
    tsne_results = tsne.fit_transform(outputs)

    # Create a DataFrame for visualization
    df = pd.DataFrame(tsne_results, columns=['x', 'y'])
    df['label'] = labels

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], c=df['label'], cmap='tab10', s=1, alpha=0.5)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(ticks=range(10), label='Classes')
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    
def plot_tsne_w_pca(outputs, labels, title, path):
    pca = PCA(n_components=5)
    outputs = pca.fit_transform(outputs)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=30, perplexity=50)
    tsne_results = tsne.fit_transform(outputs)

    # Create a DataFrame for visualization
    df = pd.DataFrame(tsne_results, columns=['x', 'y'])
    df['label'] = labels

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], c=df['label'], cmap='tab10', s=1, alpha=0.5)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(ticks=range(10), label='Classes')
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
def plot_pca(outputs, labels, title, path):
    # Run PCA
    pca = PCA(n_components=outputs.shape[1])
    if outputs.shape[1] < 2:
        raise ValueError("PCA requires at least 2 dimensions for visualization.")
    if len(outputs) == 0:
        raise ValueError("No outputs provided for PCA.")
    if len(labels) == 0:
        raise ValueError("No labels provided for PCA.")
    if len(outputs) != len(labels):
        raise ValueError("Outputs and labels must have the same length.")
    outputs = pca.fit_transform(outputs)
    singular_values = pca.singular_values_
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(singular_values) + 1), singular_values, marker='o')
    plt.title('PCA Singular Values')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path.replace('.png', '_singular_values.png'))
    plt.show()
    # Create a DataFrame for visualization
    dim_x = 0  # PC1
    dim_y = 1  # PC2

    # 5. Create DataFrame for plotting
    df = pd.DataFrame({
        'x': outputs[:, dim_x],
        'y': outputs[:, dim_y],
        'label': labels
    })
    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], c=df['label'], cmap='tab10', s=1, alpha=0.5)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(ticks=range(10), label='Classes')
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    
    
    
def extract_latent_representations(dataloader, model,device):
    """Extract latent representations from the model."""
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model.module.latent(inputs)
            all_outputs.append(outputs.cpu().numpy())
    return np.concatenate(all_outputs, axis=0)



def safe_probit(p):
    epsilon = 1e-6
    p = min(max(p, epsilon), 1 - epsilon)
    return norm.ppf(p)
def plot_probit_relationship(df_list, name_list):
    # # Extract the accuracies
    # cifar10_acc = df['cifar10_acc'].tolist()
    # cifar102_acc = df['cifar102_acc'].tolist()

    # cifar10_acc2 = df2['cifar10_acc'].tolist()
    # cifar102_acc2 = df2['cifar102_acc'].tolist()
    # Convert to proportions
    plt.figure(figsize=(8, 6))
    
    for i in range(len(df_list)):
        df = df_list[i]
        cifar10_acc = df['cifar10_acc'].tolist()
        cifar102_acc = df['cifar102_acc'].tolist()
        cifar10_probs = df['cifar10_acc'] / 100.0
        cifar102_probs = df['cifar102_acc'] / 100.0

    # Apply probit transform safely

        cifar10_probit = cifar10_probs.apply(safe_probit)
        cifar102_probit = cifar102_probs.apply(safe_probit)
        plt.plot(cifar10_probit, cifar102_probit, '.', label=name_list[i])
    # Optional: Linear Fit
    # # from sklearn.linear_model import LinearRegression
    # X = np.array(cifar10_probit).reshape(-1, 1)
    # y = np.array(cifar102_probit)
    # # reg = LinearRegression().fit(X, y)
    # # preds = reg.predict(X)
    # plt.plot(cifar10_probit, preds, 'r--', label='Linear Fit')

    # Set axis ticks to reflect accuracy percentages
    tick_accs = np.linspace(50, 95, 10)  # Adjust the range as needed
    tick_probs = [safe_probit(p / 100.0) for p in tick_accs]
    plt.xticks(tick_probs, [f"{p:.0f}%" for p in tick_accs])
    plt.yticks(tick_probs, [f"{p:.0f}%" for p in tick_accs])

    plt.xlabel("CIFAR-10 Accuracy")
    plt.ylabel("CIFAR-10.2 Accuracy")
    plt.title("Accuracy Relationship (Probit Transformed, Accuracy Axes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Accuracy Relationship (Probit Space, Accuracy Axes)")  # Save to file

    plt.show()


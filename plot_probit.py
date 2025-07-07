import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from utils import *
# Read the CSV file
# df1 = pd.read_csv("/raid/quandh/pytorch-cifar/log/cifar/accuracies.csv")  # replace with the actual path if needed
# df2 = pd.read_csv("/raid/quandh/pytorch-cifar/log/cifar2/accuracies.csv")  # replace with the actual path if needed
df1 = pd.read_csv("/raid/quandh/pytorch-cifar/log/cifar2/accuracies.csv")  # replace with the actual path if needed 
df2 = pd.read_csv("/raid/quandh/pytorch-cifar/log/cifar/accuracies.csv")  # replace with the actual path if needed


df_list = [df1, df2]
name_list = ['cifar_10.2', 'cifar_10']
# df_list = [df, df2, df4]
# df_list = [df, df2, df3]
# Preview the first few rows
# print(df.head())
# name_list = ['BCE', 'CorrBCE', "lda"]
# name_list = ['BCE', 'CorrBCE', 'cifar_10.2']

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



plot_probit_relationship(df_list, name_list)


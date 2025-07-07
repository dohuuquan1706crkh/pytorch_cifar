import torchvision
import torchvision.transforms as transforms

# Define transform (convert PIL to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts [0, 255] → [0.0, 1.0]
])

# Download training and test sets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print(f"Train set size: {len(trainset)} images")
print(f"Test set size: {len(testset)} images")

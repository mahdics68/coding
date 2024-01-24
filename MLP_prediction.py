import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from MLP_training import MLP
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm


# Define the neural network architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = MLP(input_size=784, hidden_size=128, output_size=10)
weight_path = '/scratch-grete/usr/nimmahen/models/mymodel/checkpoints.pth'
model.load_state_dict(torch.load(weight_path))
model.eval()

# Define the transformation for test data (include normalization if needed)
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST test dataset
test_dataset = datasets.MNIST(root='/scratch-grete/usr/nimmahen/data/MNIST/', train=False, transform=transform, download=True)


images_stacked = torch.stack([img for img, _ in test_dataset], dim=3)

# Calculate mean and variance along the channels dimension
mean = images_stacked.view(1, -1).mean(dim=1).item()
variance = images_stacked.view(1, -1).var(dim=1).item()

# print("Mean:", mean.item())
# print("Variance:", variance.item()) # for print: remove item

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,),(np.sqrt(variance),))]) #, 

test_dataset = datasets.MNIST(root='/scratch-grete/usr/nimmahen/data/MNIST/', transform= transform, download=True)

subset_size = 100
subset_incdices = range(subset_size)
sub_test_dataset = Subset(test_dataset, subset_incdices)

# Create a DataLoader for the test dataset
test_dataloader = DataLoader(dataset=sub_test_dataset, batch_size=1, shuffle=False)


# Specify the directory to save the prediction images
save_dir = '/scratch-grete/usr/nimmahen/models/mymodel/predictions/'
os.makedirs(save_dir, exist_ok=True)

# Iterate through the test dataset and save original and predicted images
for i, (image, label) in enumerate(tqdm(test_dataloader)):
    # Forward pass to get predictions
    with torch.no_grad():

        outputs = model(image.view(-1, 28 * 28))

    _, predicted = torch.max(outputs.data, 1)
    

    # Save the original and predicted images
    original_image_path = os.path.join(save_dir, f'original_{i}_label_{label.item()}.png')
    predicted_image_path = os.path.join(save_dir, f'predicted_{i}_label_{predicted.item()}.png')

    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Original - Label: {label.item()}, Predicted: {predicted.item()}')
    plt.savefig(original_image_path)

    # plt.imshow(outputs.squeeze().numpy(), cmap='gray')  # Assuming the output is an image
    # plt.title(f'Predicted - Label: {label.item()}, Predicted: {predicted.item()}')
    # plt.savefig(predicted_image_path)

    plt.bar(range(10), F.softmax(outputs, dim=1).squeeze().numpy())
    plt.title(f'Predicted - Label: {label.item()}, Predicted: {predicted.item()}')
    plt.savefig(predicted_image_path)


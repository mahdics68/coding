import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os
from tqdm import tqdm



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

input_size = 784 ###each pixel is a feature
hidden_size = 128 ### can change based on performance
output_size = 10 ### MNIST dataset includes 10 digits

model = MLP(input_size,hidden_size, output_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01) #changed from 0.0001

transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.5,),(0.5,))

dataset = datasets.MNIST(root='/scratch-grete/usr/nimmahen/data/MNIST/', transform= transform, download=True)



# Stack all the images to calculate mean and variance
images_stacked = torch.stack([img for img, _ in dataset], dim=3)

# Calculate mean and variance along the channels dimension
mean = images_stacked.view(1, -1).mean(dim=1).item()
variance = images_stacked.view(1, -1).var(dim=1).item()

# print("Mean:", mean.item())
# print("Variance:", variance.item()) # for print: remove item

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,),(np.sqrt(variance),))]) #, 

dataset = datasets.MNIST(root='/scratch-grete/usr/nimmahen/data/MNIST/', transform= transform, download=True)


images_stacked = torch.stack([img for img, _ in dataset], dim=3)

# Calculate mean and variance along the channels dimension
mean = images_stacked.view(1, -1).mean(dim=1).item()
variance = images_stacked.view(1, -1).var(dim=1).item()

# print("Mean:", mean)
# print("Variance:", variance) # for print: remove item



# save_path = '/home/nimmahen/code/practice/'
# for i in range(5):
#     image, lable = dataset[i]
#     plt.imshow(image.squeeze(), cmap='gray')
#     plt.title(f"lable:{lable}")

#     filename =f"{save_path}sample_{i+1}_lable{lable}.png"
#     plt.savefig(filename)

# for i in range(len(dataset)):
#     image, label = dataset[i]
#     if np.isnan(image).any() or np.isinf(image).any():
#         print(f"NAN or Inf value found in sample{i+1}")
# breakpoint()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# for x, y in train_dataset:
#     print(x.shape, y)
#     breakpoint()
# breakpoint()


batch_size = 32

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

number_iteration = 1000

total_epochs = (batch_size * number_iteration) // len(train_dataloader)
#iteration_per_epoch = number_iteration / len(train_dataloader.dataset)


# for x,y in train_dataloader:
#   print(x.shape, y.shape)
#   breakpoint()

num_epochs = 20

best_val_loss = float('inf')
patience = 3
counter = 0

# for epoch in range(num_epochs):

#   model.train()
#   for inputs, labels in train_dataloader:
#     optimizer.zero_grad()
#     outputs = model(inputs.view(-1, 28*28))
#     loss = loss_fn(outputs, labels)
#     loss.backward()
#     optimizer.step()
  
for epoch in range(int(total_epochs)):


    model.train()
    train_loss = 0
    for inputs, labels in tqdm(train_dataloader): #### 
        
        optimizer.zero_grad()
        outputs = model(inputs.view(-1, 28*28))
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    #print(f"Epoch:[{epoch+1}/{total_epochs}], Iteration:[{iteration+1}/{batches_per_iteration}] ")

    model.eval()
    val_loss= 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in val_dataloader:
            

            outputs = model(inputs.view(-1, 28*28))
            val_loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    avg_val_loss = val_loss/len(val_dataloader)
    val_acc = correct/total


    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0

    else:
        counter +=1
        if counter >= patience:
            print('Early stopping! training stopped')
            break

    print(f'Epoch [{epoch +1}/{total_epochs}], Loss: {train_loss:.4f}, Val Loss:{avg_val_loss:.4f}, Val accuracy:{val_acc:.4f}')


save_dir = '/scratch-grete/usr/nimmahen/models/mymodel/'
os.makedirs(save_dir, exist_ok=True)
file_name = 'checkpoints.pth'
weight_path = os.path.join(save_dir, file_name)

torch.save(model.state_dict(), weight_path)

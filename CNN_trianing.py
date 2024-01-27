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
from dataset_class import FlowerDataset, custom_collate



class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(-1, 128 * 28 * 28)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x




def train(model,train_dataloader, val_dataloader, optimizer, loss_fn, total_epochs, patience=30, weight_decay= 0.001):

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    counter = 0
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(int(total_epochs)):



        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            L2_regul = 0.0
            for param in model.parameters():
                L2_regul += torch.norm(param,2)

            loss = loss_fn(outputs, labels) + weight_decay*L2_regul
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
                inputs, labels = inputs.to(device), labels.to(device)
                

                outputs = model(inputs)

                val_loss += loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        avg_val_loss = val_loss/len(val_dataloader)
        val_acc = correct/total

        train_losses.append(train_loss/len(train_dataloader))
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0

        else:
            counter +=1
            if counter >= patience:
                print('Early stopping! training stopped')
                break

        print(f'Epoch [{epoch +1}/{total_epochs}], Loss: {train_loss:.4f}, Val Loss:{avg_val_loss:.4f}, Val accuracy:{val_acc:.4f}')

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Training and validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(val_accuracies, label='acc')
    plt.title('val accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    plt.savefig('/home/nimmahen/code/practice/flowers_train_val_curves.png')

    save_dir = '/scratch-grete/usr/nimmahen/models/mymodel/'
    os.makedirs(save_dir, exist_ok=True)
    file_name = 'checkpoints_flowers.pth'
    weight_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), weight_path)


def main():
    
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN(num_classes=5)
    model.to(device)

    

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.0001, weight_decay=0.001) #changed from 0.0001

    #transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.5,),(0.5,))

    class NormalizeBy255(transforms.Normalize):
        def __init__(self, mean, std, inplace=False):
            mean = tuple(x / 255.0 for x in mean)
            std = tuple(x / 255.0 for x in std)
            super(NormalizeBy255, self).__init__(mean, std, inplace)

    #dataset = datasets.MNIST(root='/scratch-grete/usr/nimmahen/data/MNIST/', transform= transform, download=True)
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), NormalizeBy255(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset = FlowerDataset(root_dir='/scratch-grete/usr/nimmahen/data/Flower/flower_photos/', transform=transform)
   



    # # Stack all the images to calculate mean and variance
    # images_stacked = torch.stack([img for img, _ in dataset], dim=3)

    # # Calculate mean and variance along the channels dimension
    # mean = images_stacked.view(1, -1).mean(dim=1).item()
    # variance = images_stacked.view(1, -1).var(dim=1).item()

    # print("Mean:", mean)
    # print("Variance:", variance) # for print: remove item
   

    # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((mean,),(np.sqrt(variance),))]) #, 

    # #dataset = datasets.MNIST(root='/scratch-grete/usr/nimmahen/data/MNIST/', transform= transform, download=True)
    # dataset = FlowerDataset(root_dir='/scratch-grete/usr/nimmahen/data/Flower/flower_photos/', transform=transform)






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
    batch_size = 16

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
   
    number_iteration = 1000
    total_epochs = (batch_size * number_iteration) // len(train_dataloader)

    

    train(model,train_dataloader, val_dataloader, optimizer, loss_fn, total_epochs, patience=30, weight_decay= 0.001)

if __name__ == "__main__":
    main()

    








  

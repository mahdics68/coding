import os 
from torchvision import transforms
import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from CNN_training import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=5)
weight_path = '/f/'
model.load_state_dict(torch.load(weight_path))
model.to(device)
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
dataset = FlowerDataset(root_dir= /fff/, transform=transform)

images_stacked = torch.stack([image for image,_ in dataset], dim=3)

mean = images_stacked.view(1,-1).mean(dim=1).item()
variance = images_stacked.view(1,-1).var(dim=1).item()

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)), transforms.Normalize((mean,),(np.sqrt(variance),))])

test_dataset = FlowerDataset(root_dir= '/fff/', transform=transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

save_dir = '/fff/'
os.mkdir(save_dir, exist_ok=True)

total = 0
correct = 0

for i, (inputs, labels) in enumerate(test_dataloader):

    with torch.no_grad():

        outputs = model(inputs)
    
    _, predicted = torch.max(outputs.data, 1)


    image_dir = os.path.join(save_dir, f'orignal_{i}_label_{labels.item()}.png')
    pred_dir = os.path.join)save_dir, f'predicted_{i}_label_{predicted.item()}.png'

    plt.imshow(inputs.squeeze(), cmap='gray')
    plt.title(f'')
    plt.savefig('/fff/')


    plt.bar(range(5), F.softmax(outputs, dim=1).squeeze().numpy())
    plt.title(f'')
    plt.savefig('/fff')
  
acc = correct/total
print(f'accuracy:{acc *100: .4f}%')

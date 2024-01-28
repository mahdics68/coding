class NormalizeBy255(transforms.Normalize):
        def __init__(self, mean, std, inplace=False):
            mean = tuple(x / 255.0 for x in mean)
            std = tuple(x / 255.0 for x in std)
            super(NormalizeBy255, self).__init__(mean, std, inplace)

#dataset = datasets.MNIST(root='/scratch-grete/usr/nimmahen/data/MNIST/', transform= transform, download=True)
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), NormalizeBy255(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

dataset = FlowerDataset(root_dir='/scratch-grete/usr/nimmahen/data/Flower/flower_photos/', transform=transform)

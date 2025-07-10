import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def prepare_dataloaders(data_dir, batch_size=32, val_split=0.1):
    # Standard image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes

    # Split dataset into train, val, test (80/10/10)
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names
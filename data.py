import torch
from torchvision import datasets, transforms

mean = (0.1307,)
std = (0.3081,)


def get_datasets(dir, batch_size, apply_transform=True):
    t_trans, tst_trans = get_transforms() if apply_transform is True \
        else get_tensor_transforms()

    num_workers = 2
    train_dataset = datasets.MNIST(
        dir, train=True, download=True, transform=t_trans)
    test_dataset = datasets.MNIST(
        dir, train=False, transform=tst_trans, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    return train_transforms, test_transforms


def get_tensor_transforms():
    print('[INFO][DATA] Getting data without transforms')
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return train_transforms, train_transforms

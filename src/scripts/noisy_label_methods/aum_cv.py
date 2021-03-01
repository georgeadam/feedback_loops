from typing import Dict

import hydra
import os
import numpy as np
import pandas as pd
import torch
import torchvision
import logging
from tqdm import tqdm

from src.utils.save import CSV_FILE
from src.utils.rand import set_seed

from omegaconf import DictConfig
from settings import ROOT_DIR

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs/aum.yaml")


def load_data(data_dir, batch_size, noise, augmentation):
    train_transform = []

    if augmentation:
        train_transform += [torchvision.transforms.Pad(4),
                            torchvision.transforms.RandomCrop(32),
                            torchvision.transforms.RandomHorizontalFlip()]

    train_transform += [torchvision.transforms.ToTensor()]
    train_transform += [torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    train_transform = torchvision.transforms.Compose(train_transform)
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

    shuffled_indices = torch.randperm(len(train_set.targets))
    train_set.targets = np.array(train_set.targets)
    train_set.data = train_set.data[shuffled_indices]
    train_set.targets = train_set.targets[shuffled_indices]
    original_targets = np.copy(train_set.targets)
    train_set.targets = corrupt_labels_uniform(train_set.targets, noise=noise)

    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                      (0.2023, 0.1994, 0.2010))])
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
    test_set.targets = np.array(test_set.targets)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=2, shuffle=False)

    data_loaders = {"train": train_loader, "test": test_loader}
    dataset_sizes = {"train": len(train_set), "test": len(test_set)}
    num_classes = len(np.unique(train_set.targets))

    return data_loaders, dataset_sizes, original_targets, num_classes


def create_optimizer(model, lr, momentum, weight_decay):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)


def create_model(name, num_classes, pretrained, device):
    model_fn = getattr(torchvision.models, name)
    model = model_fn(num_classes=num_classes, pretrained=pretrained).to(device)

    return model


def compute_margin(logits: torch.Tensor, labels: torch.Tensor):
    logits = logits.clone()
    label_logits = logits[torch.arange(len(labels)).to(labels.device), labels].clone()
    logits[torch.arange(len(labels)).to(labels.device), labels] = 0

    top_logits = torch.topk(logits, k=1, dim=1, largest=True).values.view(-1)

    return label_logits - top_logits


def train_loop_aum(model: torch.nn.Module, optimizer: torch.optim, criterion,
                   data_loader: torch.utils.data.DataLoader, dataset_size: int, device: str):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_logits = []
    epoch_labels = []

    for batch in data_loader:
        x, y = batch

        x, y = x.to(device), y.to(device)

        out = model(x)
        epoch_logits.append(out)
        loss = criterion(out, y)
        pred = torch.max(out, 1)[1]

        epoch_loss += loss * len(x)
        epoch_correct += torch.sum(pred == y).item()
        epoch_labels.append(y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = epoch_loss / float(dataset_size)
    epoch_acc = epoch_correct / float(dataset_size)
    epoch_logits = torch.cat(epoch_logits)
    epoch_labels = torch.cat(epoch_labels)

    return epoch_logits, epoch_labels, epoch_loss.item(), epoch_acc


def eval_model(model: torch.nn.Module, criterion, data_loader: torch.utils.data.DataLoader, dataset_size: int,
               device: str):
    model.eval()
    epoch_loss = 0.0
    epoch_correct = 0

    for batch in data_loader:
        x, y = batch

        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)
        pred = torch.max(out, 1)[1]

        epoch_loss += loss * len(x)
        epoch_correct += torch.sum(pred == y).item()

    epoch_loss = epoch_loss / float(dataset_size)
    epoch_acc = epoch_correct / float(dataset_size)

    return epoch_loss.item(), epoch_acc


def train_aum(model: torch.nn.Module, epochs: int, optimizer: torch.optim, criterion,
              data_loaders: Dict[str, torch.utils.data.DataLoader], dataset_sizes: Dict[str, int], device: str):
    margins = []

    for epoch in tqdm(range(epochs)):
        logits, labels, train_loss, train_acc = train_loop_aum(model, optimizer, criterion, data_loaders["train"],
                                              dataset_sizes["train"], device)
        margin = compute_margin(logits, labels)
        margins.append(margin)

        with torch.no_grad():
            test_loss, test_acc = eval_model(model, criterion, data_loaders["test"], dataset_sizes["test"], device)

        if epoch % 10:
            logging.info("Epoch: {} | Train Loss: {} | Test Loss: {} | Test Acc: {}".format(epoch, train_loss,
                                                                                            train_acc, test_loss,
                                                                                            test_acc))

    return margins, train_loss, train_acc, test_loss, test_acc


def corrupt_labels_uniform(targets: np.array, noise: float):
    targets = np.copy(targets)
    unique_targets = np.unique(targets)

    for unique_target in unique_targets:
        target_indices = np.where(targets == unique_target)[0]

        flip_indices = np.random.choice([0, 1], len(target_indices), p=[1 - noise, noise])
        flip_indices = np.where(flip_indices == 1)[0]
        replacements = np.random.choice(unique_targets, len(target_indices))

        targets[target_indices[flip_indices]] = replacements[flip_indices]

    return targets


def compute_aum_metrics(margins, corrupted_targets, original_targets):
    margins = torch.stack(margins, dim=1)
    aum = torch.mean(margins, dim=1)
    corrupted_indices = np.where(original_targets != corrupted_targets)[0]
    num_corrupted = len(corrupted_indices)

    sorted_indices = torch.argsort(aum).detach().cpu().numpy()[:len(corrupted_indices)]
    intersection = len(np.intersect1d(sorted_indices, corrupted_indices))

    print("Out of {} corrupted samples, by taking the bottom {} sorted AUMs, we recover {}/{} corrupted samples".format(
        len(corrupted_indices), len(corrupted_indices), intersection, len(corrupted_indices)))

    recall = intersection / float(num_corrupted)

    return aum, recall


@hydra.main(config_path=config_path)
def main(args: DictConfig):
    print(args.pretty())
    print("Saving to: {}".format(os.getcwd()))
    set_seed(args.misc.seed)

    data_dir = os.path.join(ROOT_DIR, args.data.data_dir)
    data_loaders, dataset_sizes, original_targets, num_classes = load_data(data_dir, args.data.batch_size,
                                                                           args.data.noise, args.data.augmentation)
    model = create_model(args.model.name, num_classes, args.model.pretrained, args.model.device)
    optimizer = create_optimizer(model, args.optim.lr, args.optim.momentum, args.optim.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    margins, train_loss, train_acc, test_loss, test_acc = train_aum(model, args.misc.epochs, optimizer, criterion,
                                                                    data_loaders, dataset_sizes, args.model.device)
    aum, recall = compute_aum_metrics(margins, data_loaders["train"].dataset.targets, original_targets)

    torch.save(aum, "aum.pt")
    csv_file_name = CSV_FILE
    data = {"train_loss": [train_loss], "train_acc": [train_acc], "test_loss": [test_loss], "recall": [recall]}
    data = pd.DataFrame(data)
    data.to_csv(csv_file_name, index=False, header=True)


if __name__ == "__main__":
    main()
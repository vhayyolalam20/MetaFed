# import torchvision
# from torchvision import transforms
# from torch.utils.data import random_split

# def get_dataset(name):
#     transform_mnist = transforms.Compose([transforms.ToTensor()])
#     transform_cifar = transforms.Compose([transforms.ToTensor()])

#     if name == 'MNIST':
#         full = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform_mnist)
#         test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)
#     else:
#         full = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_cifar)
#         test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_cifar)

#     len_full = len(full)
#     train_len = int(0.8 * len_full)
#     val_len = len_full - train_len
#     train, val = random_split(full, [train_len, val_len])
#     return train, val, test

import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, random_split, Subset

def get_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    return trainset, testset

def get_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    return trainset, testset

################################################################################################
# def get_dataset(name, iid=True, num_clients=20):
#     if name == 'MNIST':
#         trainset, testset = get_mnist()
#     elif name == 'CIFAR-10':
#         trainset, testset = get_cifar10()
#     else:
#         raise ValueError(f"Unknown dataset: {name}")

#     # Split trainset into train/val
#     train_len = int(0.8 * len(trainset))
#     val_len = len(trainset) - train_len
#     train, val = random_split(trainset, [train_len, val_len])

#     if iid:
#         user_groups = iid_partition(train, num_clients)
#     else:
#         user_groups = noniid_partition(train, num_clients)

#     return train, val, testset, user_groups

###################################################################################################
def get_dataset(args):
    name = args.dataset
    iid = args.iid
    num_clients = args.num_clients

    if name == 'MNIST':
        trainset, testset = get_mnist()
    elif name == 'CIFAR-10':
        trainset, testset = get_cifar10()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_len = int(0.8 * len(trainset))
    val_len = len(trainset) - train_len
    train, val = random_split(trainset, [train_len, val_len])

    if iid:
        user_groups = iid_partition(train, num_clients)
    else:
        user_groups = noniid_partition(train, num_clients)

    return train, val, testset, user_groups

###################################################################################################

def iid_partition(dataset, num_clients):
    data_per_client = int(len(dataset) / num_clients)
    all_idxs = np.arange(len(dataset))
    np.random.shuffle(all_idxs)
    user_groups = {i: all_idxs[i * data_per_client:(i + 1) * data_per_client] for i in range(num_clients)}
    return user_groups

def noniid_partition(dataset, num_clients, shards_per_client=2):
    num_shards = num_clients * shards_per_client
    num_imgs = len(dataset) // num_shards

    idxs = np.arange(len(dataset))
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1].argsort()]
    idxs = idxs_labels[0]

    user_groups = {i: np.array([], dtype='int64') for i in range(num_clients)}
    shard_idxs = np.array_split(idxs, num_shards)

    shard_pool = list(range(num_shards))
    for i in range(num_clients):
        rand_shards = np.random.choice(shard_pool, shards_per_client, replace=False)
        shard_pool = list(set(shard_pool) - set(rand_shards))
        for shard in rand_shards:
            user_groups[i] = np.concatenate((user_groups[i], shard_idxs[shard]), axis=0)

    return user_groups

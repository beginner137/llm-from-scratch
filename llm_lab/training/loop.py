import torch


def get_batch(dataset, batch_size, context_length, device):
    starts = torch.randint(low=0, high=(
        len(dataset) - context_length), size=(batch_size, ))
    inputs = torch.stack([
        torch.as_tensor(dataset[start: start + context_length])
        for start in starts
    ])
    targets = torch.stack([
        torch.as_tensor(dataset[start+1: start + context_length + 1])
        for start in starts
    ])
    return inputs.to(device), targets.to(device)

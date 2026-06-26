import torch

from llm_lab.training.losses import cross_entropy


def get_batch(dataset, batch_size, context_length, device):
    starts = torch.randint(low=0, high=(
        len(dataset) - context_length), size=(batch_size, ))
    inputs = torch.stack([
        torch.as_tensor(dataset[start.item(): start.item() + context_length])
        for start in starts
    ])
    targets = torch.stack([
        torch.as_tensor(
            dataset[start.item()+1: start.item() + context_length + 1])
        for start in starts
    ])
    return inputs.to(device=device, dtype=torch.long), targets.to(device=device, dtype=torch.long)


def estimate_loss(model, dataset, batch_size, context_length, device, eval_iters):
    was_training = model.training
    model.eval()

    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            inputs, targets = get_batch(
                dataset,
                batch_size,
                context_length,
                device,
            )
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            losses.append(loss.detach())

    if was_training:
        model.train()

    return torch.stack(losses).mean().item()

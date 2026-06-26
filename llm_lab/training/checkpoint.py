import torch


def save_checkpoint(model, optimizer, iteration, out):
    checkpoint_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint_dict, out)


def load_checkpoint(src, model, optimizer):
    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict["model"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    iteration = checkpoint_dict["iteration"]
    return iteration

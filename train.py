# -*- coding: utf-8 -*-
"""
Compare with the CIFAR-10 Papers With Code pareto frontier here
https://paperswithcode.com/sota/image-classification-on-cifar-10?dimension=PARAMS
"""
import functools
import click
import numpy as np
import utils
import tqdm
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from models import OriginalModel, ResidualNetwork

from torch.utils.tensorboard import SummaryWriter
from dataloaders import get_cifar10_dataloader

torch.backends.cudnn.benchmark = True

global_step = 0


def train_epoch(model, loader, optimizer, scheduler, tb_writer: SummaryWriter):
    global global_step
    epoch_start_time = time.time()

    epoch_training_num_correct = 0
    epoch_duration = time.time() - epoch_start_time
    n_images = 0
    for i, (images, labels) in enumerate(loader):
        n_images += images.shape[0]
        images = utils.to_cuda(images)
        labels = utils.to_cuda(labels)
        preds = model(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 4 == 0: # Only log every 4th iteration (reduce I/O)
            tb_writer.add_scalar("loss", loss.detach().cpu().item(), global_step)
            tb_writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], global_step)
        epoch_training_num_correct += (
            preds.detach().argmax(dim=1).eq(labels).sum().cpu().item())
        global_step += 1
    accuracy = epoch_training_num_correct / n_images
    tb_writer.add_scalar("accuracy", accuracy, global_step)
    print("train/accuracy: ", f"{accuracy:.4f}\t", end="", sep="")
    epoch_duration = time.time() - epoch_start_time
    return epoch_duration

@torch.no_grad()
def evaluate(model, dataloader, tb_writer):
    test_preds = torch.tensor([])
    n_images = 0
    epoch_testing_num_correct = 0
    epoch_testing_loss = 0

    for images, labels in dataloader:
        n_images += images.shape[0]
        images = utils.to_cuda(images)
        labels = utils.to_cuda(labels)
        preds = model(images)
        loss = F.cross_entropy(preds, labels)
        test_preds = torch.cat((test_preds, preds.cpu()), dim = 0)

        epoch_testing_loss += loss.item() * images.shape[0]
        epoch_testing_num_correct += (
            preds.argmax(dim=1).eq(labels).sum().cpu().item())

    testing_loss = epoch_testing_loss / n_images
    testing_accuracy = epoch_testing_num_correct / n_images
    print("test/loss: ", f"{testing_loss:.4f}\t ", "test/accuracy: ", f"{testing_accuracy:.4f}\t", end="", sep="")
    tb_writer.add_scalar("accuracy", testing_accuracy, global_step)
    tb_writer.add_scalar("loss", testing_loss, global_step)

model_zoo = dict(
    original=OriginalModel,
    revised=functools.partial(ResidualNetwork, start_ch=32, num_blocks_per_level=1, use_residual=False),
    revised_residual=functools.partial(ResidualNetwork, start_ch=32, num_blocks_per_level=1, use_residual=True),
    deeper=functools.partial(ResidualNetwork, start_ch=32, num_blocks_per_level=3, use_residual=True)    
)
@click.command()
@click.argument("run_name", default="original")
@click.option("--batch-size", default=128, type=int)
@click.option("--n-epochs", default=100, type=int)
@click.option("--max-lr", default=0.1, type=float)
@click.option("--model", default="original", type=click.Choice(model_zoo.keys()))
def main(run_name: str, batch_size: int, n_epochs: int, max_lr: int, model: str):
    training_loader, testing_loader = get_cifar10_dataloader(batch_size)
    network = utils.to_cuda(model_zoo[model]())
    example_inputs = utils.to_cuda(torch.randn((batch_size, 3, 32, 32)))
    # Got a 30% runtime boost with scripting the module (on a NVIDIA 1060 GPU)
    print(network)
    network = torch.jit.script(network, example_inputs=(example_inputs,))
    num_params = sum([np.prod(p.shape) for p in network.parameters()])

    print("Number of parameters: ", num_params/10**6, "M",  sep="")

    optimizer = optim.Adam(network.parameters(), lr=max_lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(training_loader),
        epochs=n_epochs)

    tb_train_writer = SummaryWriter(log_dir=f"outputs/{run_name}/train")
    tb_train_writer.add_scalar("num_params", num_params)
    tb_test_writer = SummaryWriter(log_dir=f"outputs/{run_name}/test")
    print("Starting train.")
    for i_epoch in range(n_epochs):
        print(f"epoch: {i_epoch}\t", end="")
        epoch_duration = train_epoch(network, training_loader, optimizer, scheduler, tb_train_writer)
        evaluate(network, testing_loader, tb_test_writer)
        print(f"epoch duration: {epoch_duration:.4f}")



main()

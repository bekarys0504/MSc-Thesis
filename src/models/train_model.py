import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from EEGClassification import EEGClassification
from omegaconf import OmegaConf
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import DatasetFolder


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    num_epochs = 5
    opt_func = torch.optim.Adam
    lr = 0.001
    batch_size = 1
    val_size = 2

    data_dir = input_filepath+"train/"
    test_data_dir = input_filepath+"test/"

    #load the train and test data
    dataset = DatasetFolder(
        root=data_dir,
        loader=npy_loader,
        extensions='.npy'
    )
    test_dataset = DatasetFolder(
        root=test_data_dir,
        loader=npy_loader,
        extensions='.npy'
    )

    train_size = len(dataset) - val_size 

    train_data,val_data = random_split(dataset,[train_size,val_size])
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")


    #load the train and validation into batches.
    train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)

    model = EEGClassification()
    #fitting the model on training data and record the result after each epoch
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        model.double()
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    config = OmegaConf.load('./config/config.yaml')
    main()
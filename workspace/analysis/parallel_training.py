import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, model, model_param, adam_param, dataset,
          batch_size, epochs, log_interval, fold):
    setup(rank, world_size)
    # Define model, loss function, and optimizer
    model = model(*model_param).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(ddp_model.parameters(), lr=adam_param["lr"], 
                           weight_decay=adam_param["weight_decay"])

    # Use DistributedSampler to handle distributed data loading
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    for epoch in range(epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)  # Ensure each epoch sees different data

        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f'Rank {rank}, Epoch [{epoch}/{args.epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), f"ddp_trained_model_fold_{fold}.pth")
    cleanup()


def run_train(model, model_param, adam_param, dataset,
              batch_size, epochs, log_interval, fold):
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size, model, model_param, adam_param, dataset, batch_size, epochs, log_interval, fold),
             nprocs=world_size,
             join=True)


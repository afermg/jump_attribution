import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, model, model_param, adam_param, dataset,
          batch_size, epochs, fold,
          filename, child_conn):
    setup(rank, world_size)
    # Define model, loss function, and optimizer
    model = model(*model_param).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(ddp_model.parameters(), lr=adam_param["lr"], 
                           weight_decay=adam_param["weight_decay"])

    # Use DistributedSampler to handle distributed data loading
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    for epoch in (tqdm(range(epochs)) if rank==0 else range(epochs)):
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
            del data
            del target
            torch.cuda.empty_cache()

        #Summing Loss across GPU 
        running_loss = torch.tensor(running_loss, dtype=torch.float32, device=rank)
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        running_loss_cpu = running_loss.cpu().numpy()
        del running_loss
        if rank == 0:
            child_conn.send(running_loss_cpu / len(dataset))
        
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), filename+f"{fold}.pth")
    cleanup()


def run_train(model, model_param, adam_param, dataset,
              batch_size, epochs, fold, filename):
    world_size = torch.cuda.device_count()

    parent_conn, child_conn = mp.Pipe()
    losses = []
    
    mp.spawn(train,
             args=(world_size, model, model_param, adam_param, dataset, batch_size, epochs, fold, filename, child_conn),
             nprocs=world_size,
             join=True)

    while parent_conn.poll():
        losses.append(parent_conn.recv())
    return losses


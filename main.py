import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
from config_v1 import DefaultConfig
from data.dataset_v1 import MultiSourceDataset
from models.HydroNet_v1 import HydroNet
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import logging


def setup_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger()

def relative_position(geo_coords, region_coords):
    lat_min, lon_min = region_coords[0]
    lat_max, lon_max = region_coords[1]

    rel_lat = (geo_coords[:, 0] - lat_min) / (lat_max - lat_min)
    rel_lon = (geo_coords[:, 1] - lon_min) / (lon_max - lon_min)

    return torch.stack((rel_lat, rel_lon), dim=1)

def train(rank, world_size, configs):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    use_cuda = None     # torch.cuda.is_available()
    device = torch.device(f'cuda:{rank}' if use_cuda else 'cpu')
    
    dist.init_process_group("nccl" if use_cuda else "gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, f'log_rank_{rank}.log'))
    
    data_sources = configs.data_sources
    labels = configs.labels
    cood = configs.cood
    cood = torch.tensor(cood)
    region_coords = configs.region_coords
    rel_pos = relative_position(cood, region_coords).to(device)
    dataset = MultiSourceDataset(data_sources, labels, configs)

    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    batch_size = configs.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    S01 = torch.from_numpy(dataset.data_sources['S01']).to(device)
    S02 = torch.from_numpy(dataset.data_sources['S02']).to(device)
    QS01 = torch.from_numpy(dataset.data_sources['QS01']).to(device)
    QS02 = torch.from_numpy(dataset.data_sources['QS02']).to(device)


    model = HydroNet(configs).to(device)
    model = DDP(model, device_ids=[rank] if use_cuda else None, find_unused_parameters=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    num_epochs = configs.max_epoch

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 5
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc='迭代训练中......'):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        i = 0
        for inputs, labels in train_loader:
            i += 1
            inputs, labels = inputs.to(device), labels.to(device)
            # 输出 inputs 和 labels 的值
            print(f"Rank {rank}, Epoch {epoch}, Batch {i}")
            print("Inputs:", inputs)
            print("Labels:", labels)
            
            optimizer.zero_grad()

            outputs = model([S01, S02], [QS01, QS02], inputs, rel_pos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"===========Training:{i}/{len(train_loader)}===========")

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}")
        val_loss = 0.0
        model.eval()
        i = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                i += 1
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model([S01, S02], [QS01, QS02], inputs, rel_pos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                print(f"===========validating:{i}/{len(val_loader)}===========")

        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model_rank_{rank}.pth'))
            logger.info(f"Model saved to {os.path.join(checkpoint_dir, f'best_model_rank_{rank}.pth')}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            logger.info("Early stopping")
            break

    dist.destroy_process_group()

def main():
    configs = DefaultConfig
    world_size = min(torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count(), 4)  # 使用少量的进程
    mp.spawn(train, args=(world_size, configs), nprocs=world_size, join=True)
    print('Finished Training')

if __name__ == "__main__":
    main()

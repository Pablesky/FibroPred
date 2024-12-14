from models.DatasetFibro import DatasetFibro
import torch

def create_dataset(csv_path, train_split):
    dataset = DatasetFibro(
        csv_path = csv_path
    )
    
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset, dataset

def create_dataloader(train_dataset, test_dataset, batch_size):
    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False
    )
    
    return train_dataloader, test_dataloader
    
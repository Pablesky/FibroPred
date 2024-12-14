from models.data_utils import create_dataset, create_dataloader
from models.DataEncoder import DataEncoder
import argparse

import config as cfg
import torch

def main():
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--csv_path', type=str, default='content/FibroPredCODIFICADA_Updated_after_diagnosis.csv', help='Path to the csv file')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train split')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dim_size', type=int, default=32, help='Dimension size')
    args = parser.parse_args()
        
    train_dataset, test_dataset, full_dataset = create_dataset(
        csv_path = args.csv_path,
        train_split = args.train_split,
    )
    
    train_dataloader, test_dataloader = create_dataloader(
        train_dataset=train_dataset, 
        test_dataset=test_dataset,
        batch_size=args.batch_size
    )
    
    model = DataEncoder(
        vars_per_feature = full_dataset.return_var_per_feature(),
        dim_size = args.dim_size
    )
    
    model = model.to(cfg.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()
    
    input_features = full_dataset.return_features_name()
    
    for epoch in range(args.num_epochs):
        model.train()
        
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            x = {feature: data[feature] for feature in input_features}
            
            y_hat = model(x, add_noise=True)
            y = model.get_embeebed_features(x)
            
            loss = criterion(y_hat, y)
            
            loss.backward()
                        
            optimizer.step()
            
            print(f'Train Epoch: {epoch} | Iteration: {i} | Loss: {loss.item()}')
            
        model.eval()
        
        for i, data in enumerate(test_dataloader):
            x = {feature: data[feature] for feature in input_features}
            
            y_hat = model(x)
            y = model.get_embeebed_features(x)
            
            loss = criterion(y_hat, y)
            
            print(f'Eval Epoch: {epoch} | Iteration: {i} | Loss: {loss.item()}')
    
    # Move model to cpu
    model = model.to('cpu')
    torch.save(model.state_dict(), 'weights/autoencoder.pth')
    
    
if __name__ == '__main__':
    main()
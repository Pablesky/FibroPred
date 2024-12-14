import torch
import torch.nn as nn

import config as cfg

class ReplicateLayer(nn.Module):
    def __init__(self, dim_size):
        super(ReplicateLayer, self).__init__()
        self.dim_size = dim_size

    def forward(self, x):
        # Assuming x has shape (batch_size,)
        # Replicate the scalar values in x to match the desired dim_size
        return x.expand(-1, self.dim_size).unsqueeze(1)


class DataEncoder(nn.Module):
    def __init__(self, vars_per_feature, dim_size):
        super(DataEncoder, self).__init__()
        
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=var, embedding_dim=dim_size) if var != -1 else ReplicateLayer(dim_size)
            for var in vars_per_feature
        ])
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_size, dim_size//2),
                # nn.ReLU(),
                nn.Linear(dim_size//2, dim_size//4)
            )
            for _ in vars_per_feature
        ])
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_size//4, dim_size//2),
                nn.Linear(dim_size//2, dim_size)
            )
            for _ in vars_per_feature
        ])
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
        
    def get_embeebed_features(self, x):
        with torch.no_grad():
            device = next(self.parameters()).device
            
            for feature in x.keys():
                x[feature] = x[feature].to(cfg.PRECISSION)
                x[feature] = x[feature].to(device)
                            
            embeddings = [
                embedding_layer(x[feature]).squeeze(1) for embedding_layer, feature in zip(self.embedding_layers, x.keys())
            ]
            
            embeddings = torch.stack(embeddings, dim=1)
            embeddings = embeddings.permute(1, 0, 2)
            
            return embeddings
    
    def get_space_latent(self, x):
        with torch.no_grad():
            device = next(self.parameters()).device
            
            for feature in x.keys():
                x[feature] = x[feature].to(cfg.PRECISSION)
                x[feature] = x[feature].to(device)
                            
            embeddings = [
                embedding_layer(x[feature]).squeeze(1) for embedding_layer, feature in zip(self.embedding_layers, x.keys())
            ]
            
            embeddings = torch.stack(embeddings, dim=1)
            embeddings = embeddings.permute(1, 0, 2)
            
            encoded = [
                encoder(embedding) for encoder, embedding in zip(self.encoder, embeddings)
            ]
                    
            encoded = torch.stack(encoded, dim=1)
            encoded = encoded.permute(1, 0, 2)
            
            return encoded
        
    def forward(self, x, add_noise=False):
        device = next(self.parameters()).device
        
        for feature in x.keys():
            x[feature] = x[feature].to(cfg.PRECISSION)
            x[feature] = x[feature].to(device)
                        
        embeddings = [
            embedding_layer(x[feature]).squeeze(1) for embedding_layer, feature in zip(self.embedding_layers, x.keys())
        ]
        
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = embeddings.permute(1, 0, 2)
        
        # Add noise to the embeddings
        if add_noise:
            embeddings = embeddings.clone() + torch.randn_like(embeddings) * 0.01
        
        encoded = [
            encoder(embedding) for encoder, embedding in zip(self.encoder, embeddings)
        ]
                
        encoded = torch.stack(encoded, dim=1)
        encoded = encoded.permute(1, 0, 2)
        
        decoded = [
            decoder(enc) for decoder, enc in zip(self.decoder, encoded)
        ]
        
        decoded = torch.stack(decoded, dim=1)
        decoded = decoded.permute(1, 0, 2)
                
        return decoded
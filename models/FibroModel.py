import torch
import torch.nn as nn

class ReplicateLayer(nn.Module):
    def __init__(self, dim_size):
        super(ReplicateLayer, self).__init__()
        self.dim_size = dim_size

    def forward(self, x):
        # Replicate the scalar values in x to match the desired dim_size
        return x.expand(-1, self.dim_size).unsqueeze(1)

class FibroModel(nn.Module):
    def __init__(self, vars_per_feature, dim_size, tgt_output_size):
        super(FibroModel, self).__init__()

        # Define the embedding layers for each feature
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=var, embedding_dim=dim_size) if var != -1 else ReplicateLayer(dim_size)
            for var in vars_per_feature
        ])
        
        # Encoder layers for each feature (similar to a linear transformation)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_size, dim_size//2),
                nn.ReLU(),
                nn.Linear(dim_size//2, dim_size//4)
            )
            for _ in vars_per_feature
        ])

        # Define the transformer encoder layers (3 layers)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim_size//4, nhead=4, dim_feedforward=dim_size//8)
            for _ in range(3)
        ])
        
        # Define binary classification heads (one for each transformer layer)
        self.classification_heads = nn.ModuleList([
            nn.Linear(dim_size//4, 1)  # Single output for binary classification
            for _ in range(3)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # Apply embedding layers
        embeddings = [embedding(x[:, i]) for i, embedding in enumerate(self.embedding_layers)]
        
        # Apply encoding layers to each feature embedding
        encoded = [encoder(emb) for encoder, emb in zip(self.encoder, embeddings)]
        
        # Stack encoded features to create a sequence for the transformer
        encoded_features = torch.stack(encoded, dim=1)  # Shape: (batch_size, num_features, dim_size//4)

        # Pass through transformer layers
        transformer_outputs = encoded_features
        for transformer_layer in self.transformer_layers:
            transformer_outputs = transformer_layer(transformer_outputs)

        # Apply classification heads for each transformer layer output
        output_predictions = []
        for i in range(3):
            # Each head outputs a single binary classification result
            output_predictions.append(torch.sigmoid(self.classification_heads[i](transformer_outputs[:, i, :])))
        
        return output_predictions  # List of predictions for each class


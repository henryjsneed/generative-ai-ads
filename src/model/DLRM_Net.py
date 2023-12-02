import torch
import torch.nn as nn

class DLRM_Net(nn.Module):
    def __init__(self, num_dense_features, cat_embedding_sizes):
        super(DLRM_Net, self).__init__()

        # Bottom MLP for continuous features
        self.bot_l = nn.Sequential(
            nn.Linear(num_dense_features, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.ReLU()
        )

        # Embedding layers for categorical features
        self.emb_l = nn.ModuleList([nn.Embedding(num_embeddings, 2) for num_embeddings in cat_embedding_sizes])

        # Top MLP
        self.top_l = nn.Sequential(
            nn.Linear(2 * len(cat_embedding_sizes) + 2, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_dense, x_cat):
        # Process continuous features
        x_dense = self.bot_l(x_dense)

        # Process categorical features
        x_cat = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.emb_l)]
        x_cat = torch.cat(x_cat, dim=1)

        # Combine
        x = torch.cat([x_dense, x_cat], dim=1)

        # Top MLP
        x = self.top_l(x)
        return x

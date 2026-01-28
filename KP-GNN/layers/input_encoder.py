"""
different input encoder for different dataset
"""

import torch
import torch.nn as nn


class EmbeddingEncoder(nn.Module):
    """Encoder with embedding layer
    Args:
        input_size (int): input size of feature
        hidden_size (int): hidden size of encoder
    """
    def __init__(self, input_size, hidden_size):
        super(EmbeddingEncoder, self).__init__()
        self.init_proj = nn.Embedding(input_size, hidden_size)

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self, data):
        #print(data.x)
        
        return self.init_proj(data.x.long())


class LinearEncoder(nn.Module):
    """Encoder with linear projection layer
    Args:
        input_size (int): input size of feature
        hidden_size (int): hidden size of encoder
    """
    def __init__(self, input_size, hidden_size):
        super(LinearEncoder, self).__init__()
        self.init_proj = nn.Linear(input_size, hidden_size)

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self, data):
        
        return self.init_proj(data.x.float())
    
    
class LinearKerEncoder(nn.Module):
    """Encoder with linear projection layer
    Args:
        input_size (int): input size of feature
        hidden_size (int): hidden size of encoder
        K (int, optional): number of hops. If provided, uses per-hop projections.
    """
    def __init__(self, input_size, hidden_size, K=None):
        super(LinearKerEncoder, self).__init__()
        self.K = K
        if self.K is None:
            self.init_proj = nn.Linear(input_size, hidden_size)
        else:
            # Use independent linear layers for each hop
            self.projs = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(self.K)])

    def reset_parameters(self):
        if self.K is None:
            self.init_proj.reset_parameters()
        else:
            for p in self.projs:
                p.reset_parameters()

    def forward(self, data):
        if self.K is None:
            return self.init_proj(data.kernel_emb)
        else:
            outs = []
            for i in range(self.K):
                # data.kernel_emb: [N, K, t]
                outs.append(self.projs[i](data.kernel_emb[:, i, :]))
            return torch.cat(outs, dim=1)  # [N, K * hidden_size] where hidden_size is usually dk
    
    
class LinearEntEncoder(nn.Module):
    """Encoder with linear projection layer
    Args:
        input_size (int): input size of feature (K)
        hidden_size (int): hidden size of encoder
    """
    def __init__(self, input_size, hidden_size):
        super(LinearEntEncoder, self).__init__()
        self.K = input_size
        self.hidden_size = hidden_size
        self.dk = hidden_size // input_size
        
        # Use independent linear layers for each hop to preserve hop-specific info
        self.projs = nn.ModuleList([nn.Linear(1, self.dk) for _ in range(self.K)])
        self.norm = nn.BatchNorm1d(hidden_size)

    def reset_parameters(self):
        for p in self.projs:
            p.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, data):
        # data.node_entropy: [N, K]
        outs = []
        for i in range(self.K):
            # Extract entropy for hop i
            val = data.node_entropy[:, i:i+1] # [N, 1]
            # Project to dk dimensions
            outs.append(self.projs[i](val)) # [N, dk]
        
        # Concatenate to form [N, K*dk] which matches hidden_size
        out = torch.cat(outs, dim=1)
        return self.norm(out)


class QM9InputEncoder(nn.Module):
    """Input encoder for QM9 dataset
    Args:
        hidden_size (int): hidden size of encoder
        use_pos (bool): If true, add 3D position information
    """
    def __init__(self, hidden_size, use_pos=False):
        super(QM9InputEncoder, self).__init__()
        self.use_pos = use_pos
        if use_pos:
            input_size = 22
        else:
            input_size = 19
        self.init_proj = nn.Linear(input_size, hidden_size)
        self.z_embedding = nn.Embedding(1000, 8)

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        self.z_embedding.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        if "pos" in data:
            pos = data.pos
        else:
            pos = None

        z_emb = 0
        if z is not None:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        # concatenate with continuous node features
        x = torch.cat([z_emb, x], -1)

        if self.use_pos:
            x = torch.cat([x, pos], 1)

        x = self.init_proj(x)
        return x

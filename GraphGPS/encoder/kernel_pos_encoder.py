import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder, register_edge_encoder
from torch_geometric.utils import to_dense_batch, to_dense_adj



class KernelPENodeEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding node encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = getattr(cfg, f"posenc_{self.kernel_type}")
        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        if self.kernel_type == 'VonNeumann':
            num_rw_steps = pecfg.k_order
        else:
            num_rw_steps = len(pecfg.kernel.times)
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch


@register_node_encoder('RWSE')
class RWSENodeEncoder(KernelPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'RWSE'


@register_node_encoder('HKdiagSE')
class HKdiagSENodeEncoder(KernelPENodeEncoder):
    """Heat kernel (diagonal) Structural Encoding node encoder.
    """
    kernel_type = 'HKdiagSE'


@register_node_encoder('ElstaticSE')
class ElstaticSENodeEncoder(KernelPENodeEncoder):
    """Electrostatic interactions Structural Encoding node encoder.
    """
    kernel_type = 'ElstaticSE'
    
    
@register_node_encoder('VonNeumann')  
class VonNeumannNodeEncoder(KernelPENodeEncoder):  
    """Von Neumann Entropy node encoder."""  
    kernel_type = 'VonNeumann'


class KernelPEEdgeEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding edge encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each edge feature vector.
    If `expand_edge` set True, original edge features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_emb: Size of final edge embedding
        expand_edge: Expand edge features `edge_attr` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_emb, expand_edge=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_in = 1
        if hasattr(cfg.dataset, 'edge_dim'):
            dim_in = cfg.dataset.edge_dim
        elif cfg.dataset.name in ['MNIST', 'CIFAR10']:
            dim_in = 1

        pecfg = getattr(cfg, f"posenc_{self.kernel_type}")
        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        if self.kernel_type == 'VonNeumann' or self.kernel_type == 'VonNeumannKernel':
            num_rw_steps = pecfg.k_order
        else:
            num_rw_steps = len(pecfg.kernel.times)
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_edge and dim_emb - dim_pe > 0:
            self.linear_edge = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_edge = expand_edge and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = nn.ReLU
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"pestat_edge_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        pos_enc = getattr(batch, pestat_var)  # (Num edges) x (Num kernel times)
        
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num edges) x dim_pe

        # Expand edge features if needed
        if self.expand_edge:
            h = self.linear_edge(batch.edge_attr)
        else:
            h = batch.edge_attr
        
        # Concatenate final PEs to input embedding
        batch.edge_attr = torch.cat((h, pos_enc), 1)
        
        if self.pass_as_var:
            setattr(batch, f'pe_edge_{self.kernel_type}', pos_enc)
        return batch


@register_edge_encoder('RWSE')
class RWSEEdgeEncoder(KernelPEEdgeEncoder):
    """Random Walk Structural Encoding edge encoder.
    """
    kernel_type = 'RWSE'


@register_node_encoder('VonNeumannKernelNode')
class VonNeumannKernelNodeEncoder(torch.nn.Module):
    """Von Neumann Kernel Node Encoder.
    Samples m neighbors and encodes their kernel values as node features.
    """
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in
        pecfg = cfg.posenc_VonNeumannKernel
        dim_pe = pecfg.dim_pe
        self.k_order = pecfg.k_order
        self.n_neighbors = pecfg.n_neighbors
        self.scaling = getattr(pecfg, 'scaling', 1.0)
        
        # Output dim of the raw kernel features
        # If we take mean over K, it's n_neighbors.
        # If we keep K, it's n_neighbors * k_order.
        # Following previous MPNN bias logic: mean over K -> size n_neighbors
        self.raw_feat_dim = self.n_neighbors 

        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0
        
        # Encoder for the kernel features
        self.pe_encoder = nn.Linear(self.raw_feat_dim, dim_pe)

    def forward(self, batch):
        if not hasattr(batch, 'pestat_VonNeumann'):
             raise ValueError("Precomputed 'pestat_VonNeumann' variable is required")

        # Reuse the logic for computing kernel values on the fly
        # (B, N_max, K)
        vn_h_dense, mask = to_dense_batch(batch.pestat_VonNeumann, batch.batch)

        # Optimization: Add random noise/bias to entropies to prevent 0 or identical values
        if self.training:
             vn_h_dense = vn_h_dense + torch.randn_like(vn_h_dense) * 1e-6

        B, N, K = vn_h_dense.shape

        # (B, N, N) - Adjacency including self-loops potentially if in edge_index
        # But here we want to sample neighbors.
        # Using to_dense_adj based on edge_index
        adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=N)
        
        # (B, N, N, K)
        diff = torch.abs(vn_h_dense.unsqueeze(2) - vn_h_dense.unsqueeze(1))
        if self.scaling != 1.0:
            diff = diff * self.scaling
        
        # (B, N, N) - Kernel Mean over K. 
        # Note: This is consistent with the previous MPNN bias implementation.
        kernel_val = (0.5 * torch.exp(-diff)).mean(dim=-1)

        # Sampling weights: adj + epsilon
        weights = adj.float() + 1e-6
        # Reshape to (B*N, N) for multinomial
        weights_flat = weights.view(B * N, N)

        # Sample m indices: (B*N, m)
        neighbor_indices = torch.multinomial(weights_flat, self.n_neighbors, replacement=True)

        # Gather kernel values
        kernel_flat = kernel_val.view(B * N, N)
        # (B*N, m) - The raw kernel features for each node
        sampled_kernels = torch.gather(kernel_flat, 1, neighbor_indices)
        
        # Encode
        pos_enc = self.pe_encoder(sampled_kernels) # (B*N, dim_pe)

        # Mask output to match batch.x size (remove padded nodes)
        # B*N might contain padded nodes. The active nodes are where batch.x exists.
        # But pos_enc here is flattened dense batch.
        # We need to map back to sparse batch.x
        # batch.batch tells us which graph each node belongs to, but not the dense index.
        # Actually to_dense_batch returns mask (B, N) which is True for real nodes.
        mask_flat = mask.view(-1)
        pos_enc = pos_enc[mask_flat]

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
            
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        
        return batch


@register_edge_encoder('VonNeumannKernel')
class VonNeumannKernelEdgeEncoder(KernelPEEdgeEncoder):
    """Von Neumann Kernel edge encoder.
    """
    kernel_type = 'VonNeumannKernel'

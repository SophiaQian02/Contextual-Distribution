import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_geometric.graphgym.config import cfg
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch, to_dense_adj, scatter, softmax, softmax

from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE



class VonNeumannAttention(nn.Module):
    def __init__(self, dim_h, num_heads, k_order, dim_edge=None, act='relu'):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.k_order = k_order
        self.act = register.act_dict[act]
        
        # Dimensions
        # We assume dim_h is the working dimension for nodes and edges
        
        # Node projections
        self.W_Q = nn.Linear(dim_h, dim_h)
        self.W_K = nn.Linear(dim_h, dim_h)
        self.W_V = nn.Linear(dim_h, dim_h)
        
        # Edge projections
        # Input edge dim = dim_edge (from upstream) + k_order (kernel)
        self.dim_edge_in = (dim_edge if dim_edge else dim_h) + k_order
        
        self.W_Ew = nn.Linear(self.dim_edge_in, dim_h)
        self.W_Eb = nn.Linear(self.dim_edge_in, dim_h)
        
        self.W_A = nn.Linear(dim_h, 1) # Attention scorer
        self.W_Ev = nn.Linear(dim_h, dim_h) # Edge value proj
        
        # Norm
        self.norm = nn.BatchNorm1d(dim_h) # Or LayerNorm, based on preference. Using BN as commonly used in GatedGCN.
        self.act_fn = self.act()

    def forward(self, h, edge_attr, kernel_val, batch_idx, edge_index):
        B, N, _ = to_dense_batch(h, batch_idx)[0].shape
        
        # Node Projections
        Q = self.W_Q(h)
        K = self.W_K(h) 
        V = self.W_V(h)
        
        # Prepare Edge Features
        if edge_attr is None:
             edge_feat_sparse = kernel_val
        else:
             edge_feat_sparse = torch.cat([edge_attr, kernel_val], dim=-1) # (E, Din)
        
        # Project sparse edges
        E_w = self.W_Ew(edge_feat_sparse) # (E, D)
        E_b = self.W_Eb(edge_feat_sparse)
        
        # 1. Calculate Edge Updates \hat{e}_{ij} on SPARSE edges
        # (Q_i + K_j)
        src, dst = edge_index
        fake_dense_term = Q[src] + K[dst] # (E, D)
        
        # \hat{e} = Act( Norm ( term \odot E_w + E_b ) )
        hat_e_sparse = fake_dense_term * E_w + E_b # (E, D)
        # Apply Norm? BN expects (N, C).
        # Here (E, D).
        hat_e_sparse = self.norm(hat_e_sparse)
        hat_e_sparse = self.act_fn(hat_e_sparse) # (E, D) - this is \hat{e}_{ij}
        
        # 2. Calculate Attention Scores \alpha_{ij}
        # score = W_A \hat{e}_{ij}
        alpha_sparse = self.W_A(hat_e_sparse) # (E, 1)
        
        # Softmax over neighbors (Sparse Softmax)
        # This makes it a GAT-like attention (Local).
        # If User wants GLOBAL, we need to account for non-edges.
        # Assuming non-edges have score -inf? Or 0?
        # If the formula is strictly for local neighbors (m sampled neighbors mentioned), 
        # then Sparse Softmax is correct.
        # alpha = pygnn.utils.softmax(alpha_sparse, src, ptr=None, num_nodes=h.shape[0])
        alpha = softmax(alpha_sparse, src, ptr=None, num_nodes=h.shape[0])
        
        # 3. Aggregate
        # \hat{x}_i = \sum \alpha_{ij} (W_V x_j + W_{Ev} \hat{e}_{ij})
        # V_j + Proj(\hat{e})
        msg = V[dst] + self.W_Ev(hat_e_sparse)
        weighted_msg = alpha * msg
        
        # Aggregation (Sum)
        out = scatter(weighted_msg, src, dim=0, dim_size=h.shape[0], reduce='sum')
        
        return out


class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # VonNeumannKernel Attention Bias setup
        self.vn_kernel_attn_bias = False
        if hasattr(cfg, 'posenc_VonNeumannKernel') and \
           getattr(cfg.posenc_VonNeumannKernel, 'enable', False) and \
           getattr(cfg.posenc_VonNeumannKernel, 'enable_attn_bias', False):
            self.vn_kernel_attn_bias = True
            k_order = cfg.posenc_VonNeumannKernel.k_order
            # psi: Linear projection from K-order differences to num_heads
            self.kerse_psi = nn.Linear(k_order, num_heads)
            
            # Normalization
            self.vn_normalization = getattr(cfg.posenc_VonNeumannKernel, 'normalization', '')
            self.vn_norm_module = None
            if self.vn_normalization == 'BatchNorm':
                 self.vn_norm_module = nn.BatchNorm1d(k_order)
            elif self.vn_normalization == 'LayerNorm':
                 self.vn_norm_module = nn.LayerNorm(k_order)
                 
            self.vn_scaling = getattr(cfg.posenc_VonNeumannKernel, 'scaling', 1.0)
            
        if self.vn_kernel_attn_bias:
            # Instantiate custom attention
            self.vn_attn_module = VonNeumannAttention(
                dim_h=dim_h,
                num_heads=num_heads,
                k_order=k_order,
                dim_edge=dim_h, # Assuming edge_attr is projected
                act=act
            )
            # We override global model type logic implicitly when this is active
            # or we rely on 'attn_bias' logic.
            # But based on user request, this IS the attention mechanism.
            
        # VonNeumannKernel MPNN Enhancement setup (Bias)
        self.vn_mpnn_bias = False
        if hasattr(cfg, 'posenc_VonNeumannKernel') and \
           getattr(cfg.posenc_VonNeumannKernel, 'enable', False) and \
           getattr(cfg.posenc_VonNeumannKernel, 'enable_mpnn_bias', False):
            self.vn_mpnn_bias = True
            self.vn_m_neighbors = getattr(cfg.posenc_VonNeumannKernel, 'n_neighbors', 5)
            # Projection for the m sampled kernel values
            self.vn_mpnn_proj = nn.Linear(self.vn_m_neighbors, dim_h)

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)


    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # VonNeumann MPNN Bias (Node level)
        if self.vn_mpnn_bias and hasattr(batch, 'pestat_VonNeumann'):
            # (B, N_max, K)
            vn_h_dense, mask = to_dense_batch(batch.pestat_VonNeumann, batch.batch)
            B, N, K = vn_h_dense.shape

            # (B, N, N) - Adjacency
            adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=N)

            # (B, N, N, K)
            diff = torch.abs(vn_h_dense.unsqueeze(2) - vn_h_dense.unsqueeze(1))
            if self.vn_scaling != 1.0:
                diff = diff * self.vn_scaling
            # (B, N, N) - Kernel Mean over K
            kernel_val = (0.5 * torch.exp(-diff)).mean(dim=-1)

            # Sampling weights: adj + epsilon
            weights = adj.float() + 1e-6
            weights_flat = weights.view(B * N, N)

            # Sample m indices
            # (B*N, m)
            neighbor_indices = torch.multinomial(weights_flat, self.vn_m_neighbors, replacement=True)

            # Gather kernel values
            kernel_flat = kernel_val.view(B * N, N)
            sampled_kernels = torch.gather(kernel_flat, 1, neighbor_indices)

            # Project
            h_aug_flat = self.vn_mpnn_proj(sampled_kernels)  # (B*N, dim_h)

            # Apply mask to extract valid nodes
            h_aug_dense = h_aug_flat.view(B, N, -1)
            h_aug = h_aug_dense[mask]

            h = h + h_aug

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr,
                                                   batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr)
                else:
                    h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)


        # Multi-head attention.
        if self.vn_kernel_attn_bias and hasattr(batch, 'pestat_VonNeumann'):
            # Calculate Kernel Edge Val for Attention Usage
            vn_entropy = batch.pestat_VonNeumann
            row, col = batch.edge_index
            h_u = vn_entropy[row]
            h_v = vn_entropy[col]
            diff = torch.abs(h_u - h_v)
            if self.vn_scaling != 1.0:
                diff = diff * self.vn_scaling
            kernel_edge_val = 0.5 * torch.exp(-diff) # (E, K)
            
            # Use Custom Sparse Attention
            h_attn = self.vn_attn_module(h, batch.edge_attr, kernel_edge_val, batch.batch, batch.edge_index)
            
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)
            
        elif self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            attn_bias = None
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, attn_bias, ~mask)[mask]
            elif self.global_model_type == 'BiasedTransformer':
                # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                h_attn = self._sa_block(h_dense, batch.attn_bias, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s

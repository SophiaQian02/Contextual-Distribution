import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_edge_encoder
from torch_geometric.graphgym.models.encoder import BondEncoder

from graphgps.encoder.type_dict_encoder import TypeDictEdgeEncoder
from graphgps.encoder.kernel_pos_encoder import RWSEEdgeEncoder, VonNeumannKernelEdgeEncoder

def concat_edge_encoders(encoder_classes, pe_enc_names):
    """
    A factory that creates a new Encoder class that concatenates functionality
    of the given list of two Encoder classes. First Encoder is expected
    to be a dataset-specific encoder, and the rest PE Encoders.

    Args:
        encoder_classes: List of edge encoder classes
        pe_enc_names: List of PE embedding Encoder names, used to query a dict
            with their desired PE embedding dims. That dict can only be created
            during the runtime, once the config is loaded.

    Returns:
        new edge encoder class
    """

    class Concat2EdgeEncoder(torch.nn.Module):
        """Encoder that concatenates two edge encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None

        def __init__(self, dim_emb):
            super().__init__()
            
            # PE dims can only be gathered once the cfg is loaded.
            # Note: We use 'posenc_' prefix for config lookup, similar to node encoders
            # But for edge encoders, the config might be under posenc_RWSE.enable_edge_pe
            # However, the dim_pe is usually shared or defined in the same block.
            # For VonNeumannKernel, it is in posenc_VonNeumannKernel.dim_pe
            
            # We assume the config structure follows the name
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe

            self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe)
            self.encoder2 = self.enc2_cls(dim_emb, expand_edge=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            return batch

    # Configure the correct concatenation class and return it.
    if len(encoder_classes) == 2:
        Concat2EdgeEncoder.enc1_cls = encoder_classes[0]
        Concat2EdgeEncoder.enc2_cls = encoder_classes[1]
        Concat2EdgeEncoder.enc2_name = pe_enc_names[0]
        return Concat2EdgeEncoder
    else:
        raise ValueError(f"Does not support concatenation of "
                         f"{len(encoder_classes)} encoder classes.")

# Dataset-specific edge encoders.
ds_encs = {'TypeDictEdge': TypeDictEdgeEncoder, 'Bond': BondEncoder}

# Positional Encoding edge encoders.
pe_encs = {'RWSE': RWSEEdgeEncoder,
           'VonNeumannKernel': VonNeumannKernelEdgeEncoder}

# Concat dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_edge_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_edge_encoders([ds_enc_cls, pe_enc_cls],
                                 [pe_enc_name])
        )

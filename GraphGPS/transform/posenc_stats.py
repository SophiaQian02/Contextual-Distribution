from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter, k_hop_subgraph)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from graphgps.encoder.graphormer_encoder import graphormer_pre_processing

from torch_geometric.utils import k_hop_subgraph, to_dense_adj

def get_von_neumann_entropy(edge_index, num_nodes, k_order=2, eps=1e-10):
    all_entropies = []
    for node in range(num_nodes):
        node_entropies = []
        for k in range(1, k_order + 1):
            subset, sub_edge_index, _, _, = k_hop_subgraph(
                node, k, edge_index, relabel_nodes=True, num_nodes=num_nodes)
            if subset.numel() == 0:
                node_entropies.append(0.0)
                continue
            adj = to_dense_adj(sub_edge_index, max_num_nodes=subset.size(0))[0]
            deg = adj.sum(dim=-1)
            deg_inv_sqrt = torch.diag(torch.clamp(deg, min=eps).rsqrt())
            lap = torch.eye(adj.size(0), device=adj.device) - deg_inv_sqrt @ adj @ deg_inv_sqrt
            trace = torch.clamp(lap.trace(), min=eps)
            rho = lap / trace
            vals = torch.linalg.eigvalsh(rho).clamp(min=eps)
            entropy = -(vals * vals.log()).sum()
            node_entropies.append(entropy)
        all_entropies.append(torch.tensor(node_entropies, device=edge_index.device))
    return torch.stack(all_entropies)  # shape (num_nodes, k_order)


def get_von_neumann_kernel(edge_index, num_nodes, k_order=2, eps=1e-10):
    """
    Computes the Von Neumann Kernel for edges based on node entropies.
    P_uv = (1/K) * sum_{k=1}^K (1/2 * exp(-|H(S_u^k) - H(S_v^k)|))
    """
    # First compute node entropies for all k
    node_entropies = get_von_neumann_entropy(edge_index, num_nodes, k_order, eps) # (N, K)
    
    # For each edge (u, v), compute the kernel value
    row, col = edge_index
    
    # Get entropies for source and target nodes
    h_u = node_entropies[row] # (E, K)
    h_v = node_entropies[col] # (E, K)
    
    # Compute kernel values per k
    # K(S_u^k, S_v^k) = 0.5 * exp(-|H(S_u^k) - H(S_v^k)|)
    diff = torch.abs(h_u - h_v)
    kernel_vals = 0.5 * torch.exp(-diff) # (E, K)
    
    # Return full K-dim features instead of mean
    # avg_kernel = kernel_vals.mean(dim=1, keepdim=True) # (E, 1)
    
    return kernel_vals


def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.
    'Graphormer': Computes spatial types and optionally edges along shortest paths.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE',
                     'HKfullPE', 'ElstaticSE', 'GraphormerBias', 'VonNeumann', 'VonNeumannKernel']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    laplacian_norm_type = cfg.posenc_LapPE.eigen.laplacian_norm.lower()
    if laplacian_norm_type == 'none':
        laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())
        
        if 'LapPE' in pe_types:
            max_freqs=cfg.posenc_LapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm
        elif 'EquivStableLapPE' in pe_types:  
            max_freqs=cfg.posenc_EquivStableLapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_EquivStableLapPE.eigen.eigvec_norm
        
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)

    if 'SignNet' in pe_types:
        # Eigen-decomposition with numpy for SignNet.
        norm_type = cfg.posenc_SignNet.eigen.laplacian_norm.lower()
        if norm_type == 'none':
            norm_type = None
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=norm_type,
                           num_nodes=N)
        )
        evals_sn, evects_sn = np.linalg.eigh(L.toarray())
        data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
            evals=evals_sn, evects=evects_sn,
            max_freqs=cfg.posenc_SignNet.eigen.max_freqs,
            eigvec_norm=cfg.posenc_SignNet.eigen.eigvec_norm)

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing
        
        if cfg.posenc_RWSE.enable_edge_pe:
            rw_edge = get_rw_edge_probs(ksteps=kernel_param.times,
                                        edge_index=data.edge_index,
                                        num_nodes=N)
            data.pestat_edge_RWSE = rw_edge

    # Heat Kernels.
    if 'HKdiagSE' in pe_types or 'HKfullPE' in pe_types:
        # Get the eigenvalues and eigenvectors of the regular Laplacian,
        # if they have not yet been computed for 'eigen'.
        if laplacian_norm_type is not None or evals is None or evects is None:
            L_heat = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization=None, num_nodes=N)
            )
            evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        else:
            evals_heat, evects_heat = evals, evects
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)

        # Get the full heat kernels.
        if 'HKfullPE' in pe_types:
            # The heat kernels can't be stored in the Data object without
            # additional padding because in PyG's collation of the graphs the
            # sizes of tensors must match except in dimension 0. Do this when
            # the full heat kernels are actually used downstream by an Encoder.
            raise NotImplementedError()
            # heat_kernels, hk_diag = get_heat_kernels(evects_heat, evals_heat,
            #                                   kernel_times=kernel_param.times)
            # data.pestat_HKdiagSE = hk_diag
        # Get heat kernel diagonals in more efficient way.
        if 'HKdiagSE' in pe_types:
            kernel_param = cfg.posenc_HKdiagSE.kernel
            if len(kernel_param.times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            hk_diag = get_heat_kernels_diag(evects_heat, evals_heat,
                                            kernel_times=kernel_param.times,
                                            space_dim=0)
            data.pestat_HKdiagSE = hk_diag

    # Electrostatic interaction inspired kernel.
    if 'ElstaticSE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        data.pestat_ElstaticSE = elstatic

    if 'GraphormerBias' in pe_types:
        data = graphormer_pre_processing(
            data,
            cfg.posenc_GraphormerBias.num_spatial_types
        )
        
    if 'VonNeumann' in pe_types:  
        k_order = getattr(cfg.posenc_VonNeumann, 'k_order', 3)
        # Check if already computed for this k_order
        if hasattr(data, 'pestat_VonNeumann') and hasattr(data, 'pestat_VonNeumann_k') and data.pestat_VonNeumann_k == k_order:
            pass # Already computed
        else:
            vn_entropy = get_von_neumann_entropy(  
                edge_index=data.edge_index,  
                num_nodes=data.num_nodes,  
                k_order=k_order  
            )  
            data.pestat_VonNeumann = vn_entropy
            data.pestat_VonNeumann_k = k_order # Store k to check later
            
    if 'VonNeumannKernel' in pe_types:
        k_order = getattr(cfg.posenc_VonNeumannKernel, 'k_order', 3)
        # We need node entropies for the Kernel
        # Also save them because the Encoder might need them
        vn_entropy = get_von_neumann_entropy(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            k_order=k_order
        )
        data.pestat_VonNeumann = vn_entropy
        
        # Now compute kernel from these entropies
        # Re-implementing get_von_neumann_kernel logic inline to avoid re-computation
        row, col = data.edge_index
        h_u = vn_entropy[row]

    # Single unified block for VonNeumannKernel which needs Node Entropy anyway
    # This prevents duplication and ensures "pestat_VonNeumann" exists if "VonNeumannKernel" is requested
    if 'VonNeumannKernel' in pe_types or 'VonNeumann' in pe_types:
        # Determine K
        k_vn = 3
        if 'VonNeumannKernel' in pe_types:
             k_vn = getattr(cfg.posenc_VonNeumannKernel, 'k_order', 3)
        elif 'VonNeumann' in pe_types:
             k_vn = getattr(cfg.posenc_VonNeumann, 'k_order', 3)
             
        # Check if we need to compute Node Entropy
        if hasattr(data, 'pestat_VonNeumann') and hasattr(data, 'pestat_VonNeumann_k') and data.pestat_VonNeumann_k == k_vn:
             vn_entropy = data.pestat_VonNeumann
        else:
             vn_entropy = get_von_neumann_entropy(data.edge_index, data.num_nodes, k_vn)
             data.pestat_VonNeumann = vn_entropy
             data.pestat_VonNeumann_k = k_vn

        # Compute Kernel if requested
        if 'VonNeumannKernel' in pe_types:
             row, col = data.edge_index
             h_u = vn_entropy[row]
             h_v = vn_entropy[col]
             diff = torch.abs(h_u - h_v)
             vn_kernel = 0.5 * torch.exp(-diff)
             data.pestat_edge_VonNeumannKernel = vn_kernel

    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_rw_edge_probs(ksteps, edge_index, edge_weight=None, num_nodes=None):
    """Compute Random Walk probabilities for edges for given list of K steps.
    
    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        
    Returns:
        2D Tensor with shape (num_edges, len(ksteps)) with RW edge probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        return torch.zeros((0, len(ksteps)), device=edge_index.device)
    
    # P = D^-1 * A
    P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes) # 1 x N x N
    
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            # Extract values at edge indices
            val = Pk[0, source, dest] # Shape (E,)
            rws.append(val)
            Pk = Pk @ P
    else:
        for k in ksteps:
            Pk = P.matrix_power(k)
            val = Pk[0, source, dest]
            rws.append(val)
            
    rw_edge = torch.stack(rws, dim=1) # (Num edges) x (K steps)
    return rw_edge


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.max(dim=0)[0],  # Max of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.max(dim=0)[0],  # Max of Vj -> i
        electrostatic.mean(dim=1),  # Mean of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs

"""
Utils for defining model layers
"""
from layers.KPGCN import *
from layers.KPGIN import *
from layers.KEGIN import *
from layers.KAGIN import *
from layers.KEGCN import *
from layers.KAGCN import *
from layers.KPGINplus import *
from layers.KPGraphSAGE import *
from layers.KAGraphSAGE import *
from layers.KEGraphSAGE import *


def make_gnn_layer(args):
    """function to construct gnn layer
    Args:
        args (argparser): arguments list
    """
    model_name = args.model_name
    if model_name == "KPGCN":
        gnn_layer = KPGCNConv(args.hidden_size, args.hidden_size, args.K, args.num_hop1_edge, args.max_pe_num,
                              args.combine, args.t)
    elif model_name == "KEGCN":
        gnn_layer = KEGCNConv(args.hidden_size, args.hidden_size, args.K, args.num_hop1_edge, args.max_pe_num,
                              args.combine, args.t)
    elif model_name == "KPGIN" or model_name == "KPGINPrime":
        gnn_layer = KPGINConv(args.hidden_size, args.hidden_size, args.K, args.eps, args.train_eps, args.num_hop1_edge,
                              args.max_pe_num, args.combine, args.t)
    elif model_name == "KEGIN" or model_name == "KEGINPrime" or model_name == "KEGINPlus":
        gnn_layer = KEGINConv(args.hidden_size, args.hidden_size, args.K, args.eps, args.train_eps, args.num_hop1_edge,
                              args.max_pe_num, args.combine, args.t)
    elif model_name == "KAGIN":
        gnn_layer = KAGINConv(args.hidden_size, args.hidden_size, args.K, args.eps, args.train_eps, args.num_hop1_edge,
                              args.max_pe_num, args.combine, args.t)
    elif model_name == "KAGCN":
        gnn_layer = KAGCNConv(args.hidden_size, args.hidden_size, args.K, args.num_hop1_edge,
                              args.max_pe_num, args.combine, args.t)
    elif model_name == "KPGraphSAGE":
        gnn_layer = KPGraphSAGEConv(args.hidden_size, args.hidden_size, args.K, args.aggr, args.num_hop1_edge,
                                    args.max_pe_num, args.combine, args.t)
    elif model_name == "KAGraphSAGE":
        gnn_layer = KAGraphSAGEConv(args.hidden_size, args.hidden_size, args.K, args.aggr, args.num_hop1_edge,
                                    args.max_pe_num, args.combine, args.t)
    elif model_name == "KEGraphSAGE":
        gnn_layer = KEGraphSAGEConv(args.hidden_size, args.hidden_size, args.K, args.aggr, args.num_hop1_edge,
                                    args.max_pe_num, args.combine, args.t)
    elif model_name == "KPGINPlus":
        gnn_layer = [
            KPGINPlusConv(args.hidden_size, args.hidden_size, l, args.num_hop1_edge, args.max_pe_num, args.combine, args.t)
            if l <= args.K else KPGINPlusConv(args.hidden_size, args.hidden_size, args.K, args.num_hop1_edge,
                                              args.max_pe_num, args.combine, args.t)
            for l in range(1, args.num_layer + 1)]
    else:
        raise ValueError("Not supported GNN type")

    return gnn_layer

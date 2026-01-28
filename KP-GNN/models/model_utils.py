"""
Model utils file
"""

from models.GNNs import *


def make_GNN(args):
    if args.model_name == "KPGINPlus" :
        return GNNPlus
    elif args.model_name == "KPGINPlus":
        return GNNEPlus
    elif args.model_name == "KPGINPrime" or args.model_name == "KEGINPrime":
        return GNNPrime
    # elif args.model_name == "KEGIN":
    #     return GNNE
    else:
        return GNN

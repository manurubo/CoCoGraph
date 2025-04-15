from lib_functions.libraries import *
from lib_functions.config import *
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import  GINEConv, global_max_pool, global_mean_pool

from typing import  Optional, Union
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch import Tensor

class EnhancedGINEConv(GINEConv):
    """
    An enhanced Graph Isomorphism Network (GIN) convolution layer that incorporates
    global graph features into the node update mechanism.

    It extends the standard GINEConv by adding a transformation for global features
    and including them in the final node representation calculation.

    Args:
        nn (torch.nn.Module): A neural network `h_\theta` that maps node features
            `x_j` of shape `[-1, in_channels]` to shape `[-1, out_channels]`, *e.g.*,
            defined by `torch.nn.Sequential`.
        global_feat_dim (int): Dimensionality of the global features vector.
        eps (float, optional): Initial `\epsilon` value. (default: `0.`)
        train_eps (bool, optional): If set to `True`, `\epsilon` will be a trainable
            parameter. (default: `False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to `None`,
            input edge features are not used. (default: `None`)
        **kwargs (optional): Additional arguments of
            `torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: torch.nn.Module, global_feat_dim: int,  eps: float = 0., train_eps: bool = False, edge_dim: Optional[int] = None, **kwargs):
        super().__init__(nn, eps, train_eps, edge_dim, **kwargs)
        # Transform global features to match node features' dimensionality
        if global_feat_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.global_features_transform = Linear(global_feat_dim, in_channels)
    
    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        global_features: Tensor,  # Additional argument to pass precomputed global features
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """
        Forward pass for the EnhancedGINEConv layer.

        Args:
            x (Union[Tensor, OptPairTensor]): The input node features. Can be a
                single tensor or a tuple of tensors for bipartite graphs.
            edge_index (Adj): The edge indices.
            global_features (Tensor): Precomputed global features for the graph(s).
            edge_attr (OptTensor, optional): Edge features. (default: `None`)
            size (Size, optional): The size of the output tensor. (default: `None`)

        Returns:
            Tensor: The updated node embeddings.
        """
        if isinstance(x, Tensor):
            x = (x, x)

        # Transform global features ESTO HAY QUE MIRARLO
        transformed_global_features = self.global_features_transform(global_features)
        
        # # Broadcast global features to each node
        transformed_global_features = transformed_global_features.repeat(x[1].size(0), 1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r + transformed_global_features

        return self.nn(out)


class GINEdgeQuadrupletPredictor(torch.nn.Module):
    """
    Predicts probabilities for breaking and making edges between node pairs
    using an Enhanced GINE architecture.

    It processes node features, edge features (including DOSD distances),
    global graph features, pairwise distances, and a noise level.
    The model outputs separate logit matrices for breaking and making edges,
    and calculates quadruplet probabilities based on these.

    Inputs in `forward` (via `data` object):
        - `data.x`: Node features [num_nodes, num_node_features]
        - `data.edge_index`: Graph connectivity [2, num_edges]
        - `data.edge_attr`: Edge features [num_edges, num_edge_features]
        - `data.xA`: Global graph features [num_graphs, num_global_features]
        - `data.noiselevel`: Noise level scalar for the graph.
        - `data.distances`: Pairwise distances between nodes [MAX_ATOM, MAX_ATOM, num_distance_features]
        - `data.dosd_distances`: DOSD distances between nodes [MAX_ATOM, MAX_ATOM]
        - `data.batch`: Batch assignment vector [num_nodes]

    Outputs in `forward`:
        - `pairs_break`: Logits for breaking an edge [batch_size, MAX_ATOM, MAX_ATOM]
        - `pairs_make`: Logits for making an edge [batch_size, MAX_ATOM, MAX_ATOM]
        - `quadruplet_probabilities`: Probabilities for quadruplet interactions
          [batch_size, MAX_ATOM, MAX_ATOM, MAX_ATOM, MAX_ATOM] (derived from break/make logits)
    """
    def __init__(self):
        super().__init__()
        nn1 = Sequential(Linear(NNFEAT+1, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv1 = EnhancedGINEConv(nn1, edge_dim=18, global_feat_dim= 21)
        self.ff1 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)
        nn2 = Sequential(Linear(2*NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv2 = EnhancedGINEConv(nn2, edge_dim=18, global_feat_dim= 21)
        self.ff2 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)
        nn3 = Sequential(Linear(2*NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv3 = EnhancedGINEConv(nn3, edge_dim=18, global_feat_dim= 21)
        self.ff3 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)
        
        self.noise_mlp = nn.Sequential(
            nn.Linear(1, 4),
            nn.GELU(),
            nn.Linear(4, 1),
        )
        
        self.mlpGraph = nn.Sequential(
            Linear(NGFEAT, 2*NGFEAT),
            nn.ReLU(),
            Linear(2*NGFEAT, 2*NGFEAT)
        )
        
        self.ff_break = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1),
                Linear(4*NNFEAT+2*NGFEAT+12+17+1, 256),
                nn.ReLU(),
                Linear(256, 4*NNFEAT+2*NGFEAT+12+17+1)
            ),
            nn.Sequential(
                nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1),
                Linear(4*NNFEAT+2*NGFEAT+12+17+1, 256),
                nn.ReLU(),
                Linear(256, 4*NNFEAT+2*NGFEAT+12+17+1)
            )
        ])

        self.ff_make = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1),
                Linear(4*NNFEAT+2*NGFEAT+12+17+1, 256),
                nn.ReLU(),
                Linear(256, 4*NNFEAT+2*NGFEAT+12+17+1)
            ),
            nn.Sequential(
                nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1),
                Linear(4*NNFEAT+2*NGFEAT+12+17+1, 256),
                nn.ReLU(),
                Linear(256, 4*NNFEAT+2*NGFEAT+12+17+1)
            )
        ])

        self.final_layer_norm_break = nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1)
        self.final_layer_norm_make = nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1)

        self.reducer_make = nn.Sequential(
            Linear(4*NNFEAT+2*NGFEAT+12+17+1, 1),
        )

        self.reducer_break = nn.Sequential(
            Linear(4*NNFEAT+2*NGFEAT+12+17+1, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, xA, noiselevel, distances, dosd_distances = data.x, data.edge_index, data.edge_attr, data.xA, data.noiselevel, data.distances, data.dosd_distances
        batch = data.batch

        # Add DOSD distances to edge attributes
        dosd_values = dosd_distances[edge_index[0], edge_index[1]].unsqueeze(1)
        edge_attr_added = torch.cat([edge_attr, dosd_values], dim=1)
        
        # Process noise level
        noise = self.noise_mlp(noiselevel.float().unsqueeze(0)).expand(x.size(0), -1)
        x = torch.cat([x, noise], dim=1)  # Add noise to x as a new feature
        
        # GATN part
        x = F.relu(self.conv1(x, edge_index, xA, edge_attr_added))
        x = self.ff1(x)
        x = F.relu(x)
        
        x = F.relu(self.conv2(x, edge_index, xA, edge_attr_added))
        x = self.ff2(x)
        x = F.relu(x)
        
        x = F.relu(self.conv3(x, edge_index, xA, edge_attr_added))
        x_base = self.ff3(x)
        x_gen = global_max_pool(x_base, batch)

        # ANN for the graph embedding
        shBatch = x_gen.shape[0]
        xA_m = xA.reshape([shBatch, NGFEAT])
        xann = self.mlpGraph(xA_m)

        # Create all possible pairs of embeddings
        x1 = x_base.unsqueeze(1).repeat(1, x_base.size(0), 1)  # Tamaño: [35, 35, 60]
        x2 = x_base.unsqueeze(0).repeat(x_base.size(0), 1, 1)  # Tamaño: [35, 35, 60]

        expanded_xann = xann.unsqueeze(0).expand(x1.size(0), x1.size(1), -1)    

        # Initialize the zero matrix for edge attributes
        edge_attr_matrix = torch.zeros(MAX_ATOM, MAX_ATOM, edge_attr.size(1), device=x.device)

        # Assign edge attributes using advanced indexing
        edge_attr_matrix[edge_index[0], edge_index[1]] = edge_attr
        
        # Concatenate edge attributes to pairs
        pairs = torch.cat([x1, x2, expanded_xann, distances, edge_attr_matrix, dosd_distances.unsqueeze(-1)], dim=2)  

        pairs_flattened = pairs.view(-1, MAX_ATOM * MAX_ATOM, 4*NNFEAT+2*NGFEAT+12+17+1)

        # Process with residual connections
        x_break = pairs_flattened
        for layer in self.ff_break:
            x_break = layer(x_break) + x_break  # residual connection
        pairs_break = self.final_layer_norm_break(x_break)

        x_make = pairs_flattened
        for layer in self.ff_make:
            x_make = layer(x_make) + x_make  # residual connection
        pairs_make = self.final_layer_norm_make(x_make)

        # Continue with existing code
        pairs_break = self.reducer_break(pairs_break)
        pairs_make = self.reducer_make(pairs_make)
        
        pairs_break = pairs_break.view(-1,MAX_ATOM,MAX_ATOM)
        pairs_make = pairs_make.view(-1,MAX_ATOM,MAX_ATOM)
        
        # Calculate quadruplet probabilities
        quadruplet_probabilities = self.calculate_quadruplet_probabilities(pairs_break, pairs_make)
        
        return pairs_break, pairs_make, quadruplet_probabilities
        
    def calculate_quadruplet_probabilities(self, pairs_break, pairs_make):
        """
        Calculates quadruplet interaction probabilities from break/make logits.

        Ensures symmetry and combines probabilities assuming independence.

        Args:
            pairs_break (Tensor): Logits for breaking edges [MAX_ATOM, MAX_ATOM].
            pairs_make (Tensor): Logits for making edges [MAX_ATOM, MAX_ATOM].

        Returns:
            Tensor: Sigmoid probabilities for quadruplet interactions
                    [MAX_ATOM, MAX_ATOM, MAX_ATOM, MAX_ATOM].
        """
        pairs_break = pairs_break.squeeze(0)
        pairs_make = pairs_make.squeeze(0)
        
#         # Make the probability matrices symmetric
        pairs_break = (pairs_break + pairs_break.T) / 2
        pairs_make = (pairs_make + pairs_make.T) / 2

        # Combine the probabilities of destroy and create
        quadruplet_probabilities = pairs_break.unsqueeze(2).unsqueeze(3) + pairs_break.unsqueeze(0).unsqueeze(1) + pairs_make.unsqueeze(1).unsqueeze(3) + pairs_make.unsqueeze(0).unsqueeze(2)

        quadruplet_probabilities = torch.sigmoid(quadruplet_probabilities)
        
        return quadruplet_probabilities

class GINEdgeQuadrupletPredictor_MorganFP(torch.nn.Module):
    """
    Predicts probabilities for breaking and making edges using an Enhanced GINE
    architecture, incorporating Morgan fingerprints.

    Similar to `GINEdgeQuadrupletPredictor`, but adds processed Morgan
    fingerprints to the feature representation before the final reduction layers
    for edge prediction.

    Inputs in `forward` (via `data` object):
        - All inputs from `GINEdgeQuadrupletPredictor`
        - `data.morgan_fp`: Morgan fingerprints for the graph [num_graphs, morgan_fp_dim]

    Outputs in `forward`:
        - Same as `GINEdgeQuadrupletPredictor`.
    """
    def __init__(self):
        super().__init__()
        nn1 = Sequential(Linear(NNFEAT+1, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv1 = EnhancedGINEConv(nn1, edge_dim=18, global_feat_dim= 21)
        self.ff1 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)  # Adjusted for concatenated morgan_fp
        nn2 = Sequential(Linear(2*NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv2 = EnhancedGINEConv(nn2, edge_dim=18, global_feat_dim= 21)
        self.ff2 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)  # Adjusted for concatenated morgan_fp
        nn3 = Sequential(Linear(2*NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv3 = EnhancedGINEConv(nn3, edge_dim=18, global_feat_dim= 21)
        self.ff3 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)  # Adjusted for concatenated morgan_fp
        
        self.noise_mlp = nn.Sequential(
            nn.Linear(1, 4),
            nn.GELU(),
            nn.Linear(4, 1),
        )
        
        self.mlpGraph = nn.Sequential(
            Linear(NGFEAT, 2*NGFEAT),
            nn.ReLU(),
            Linear(2*NGFEAT, 2*NGFEAT)
        )
        
        self.morgan_fp_mlp1 = nn.Sequential(
            Linear(2048, 512),
            nn.ReLU(),
            Linear(512, 256),
            nn.ReLU(),
            Linear(256, 256)
        )

        self.morgan_fp_mlp2 = nn.Sequential(
            Linear(2048, 512),
            nn.ReLU(),
            Linear(512, 256),
            nn.ReLU(),
            Linear(256, 256)
        )

        # Improved feedforward networks with LayerNorm and residual connections
        self.ff_break = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1),
                Linear(4*NNFEAT+2*NGFEAT+12+17+1, 256),
                nn.ReLU(),
                Linear(256, 4*NNFEAT+2*NGFEAT+12+17+1)
            ),
            nn.Sequential(
                nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1),
                Linear(4*NNFEAT+2*NGFEAT+12+17+1, 256),
                nn.ReLU(),
                Linear(256, 4*NNFEAT+2*NGFEAT+12+17+1)
            )
        ])

        self.ff_make = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1),
                Linear(4*NNFEAT+2*NGFEAT+12+17+1, 256),
                nn.ReLU(),
                Linear(256, 4*NNFEAT+2*NGFEAT+12+17+1)
            ),
            nn.Sequential(
                nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1),
                Linear(4*NNFEAT+2*NGFEAT+12+17+1, 256),
                nn.ReLU(),
                Linear(256, 4*NNFEAT+2*NGFEAT+12+17+1)
            )
        ])

        self.final_layer_norm_break = nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1)
        self.final_layer_norm_make = nn.LayerNorm(4*NNFEAT+2*NGFEAT+12+17+1)

        self.pre_reducer_make = nn.Sequential(
            Linear(4*NNFEAT+2*NGFEAT+12+17+1+256, 4*NNFEAT+2*NGFEAT+12+17+1),
        )

        self.pre_reducer_break = nn.Sequential(
            Linear(4*NNFEAT+2*NGFEAT+12+17+1+256, 4*NNFEAT+2*NGFEAT+12+17+1),
        )

        self.reducer_make = nn.Sequential(
            Linear(4*NNFEAT+2*NGFEAT+12+17+1, 1),
        )

        self.reducer_break = nn.Sequential(
            Linear(4*NNFEAT+2*NGFEAT+12+17+1, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, xA, noiselevel, distances, dosd_distances, morgan_fp = data.x, data.edge_index, data.edge_attr, data.xA, data.noiselevel, data.distances, data.dosd_distances, data.morgan_fp
        batch = data.batch

        # Add DOSD distances to edge attributes
        dosd_values = dosd_distances[edge_index[0], edge_index[1]].unsqueeze(1)
        edge_attr_added = torch.cat([edge_attr, dosd_values], dim=1)
        
        # Process noise level
        noise = self.noise_mlp(noiselevel.float().unsqueeze(0)).expand(x.size(0), -1)
        x = torch.cat([x, noise], dim=1)  # Add noise to x as a new feature
        
        # GATN part
        x = F.relu(self.conv1(x, edge_index, xA, edge_attr_added))
        x = self.ff1(x)
        x = F.relu(x)
        
        x = F.relu(self.conv2(x, edge_index, xA, edge_attr_added))
        x = self.ff2(x)
        x = F.relu(x)
        
        x = F.relu(self.conv3(x, edge_index, xA, edge_attr_added))
        x_base = self.ff3(x)
        x_gen = global_max_pool(x_base, batch)

        shBatch = x_gen.shape[0]
        xA_m = xA.reshape([shBatch, NGFEAT])
        xann = self.mlpGraph(xA_m)

        # Create all possible pairs of embeddings
        x1 = x_base.unsqueeze(1).repeat(1, x_base.size(0), 1)  # Tamaño: [35, 35, 60]
        x2 = x_base.unsqueeze(0).repeat(x_base.size(0), 1, 1)  # Tamaño: [35, 35, 60]

        expanded_xann = xann.unsqueeze(0).expand(x1.size(0), x1.size(1), -1)    

        # Initialize the zero matrix for edge attributes
        edge_attr_matrix = torch.zeros(MAX_ATOM, MAX_ATOM, edge_attr.size(1), device=x.device)

        # Assign edge attributes using advanced indexing
        edge_attr_matrix[edge_index[0], edge_index[1]] = edge_attr
        
        # Concatenate edge attributes to pairs
        pairs = torch.cat([x1, x2, expanded_xann, distances, edge_attr_matrix, dosd_distances.unsqueeze(-1)], dim=2)  # Adjust dimension as per actual sizes

        pairs_flattened = pairs.view(-1, MAX_ATOM * MAX_ATOM, 4*NNFEAT+2*NGFEAT+12+17+1)

        # Process with residual connections
        x_break = pairs_flattened
        for layer in self.ff_break:
            x_break = layer(x_break) + x_break  # residual connection
        pairs_break = self.final_layer_norm_break(x_break)

        x_make = pairs_flattened
        for layer in self.ff_make:
            x_make = layer(x_make) + x_make  # residual connection
        pairs_make = self.final_layer_norm_make(x_make)

        # Continue with existing code
        morgan_fp_break = self.morgan_fp_mlp1(morgan_fp.float())
        morgan_fp_make = self.morgan_fp_mlp2(morgan_fp.float())
        pairs_break = torch.cat([pairs_break, morgan_fp_break.unsqueeze(0).unsqueeze(0).expand(pairs_break.size(0), pairs_break.size(1), -1)], dim=2)
        pairs_make = torch.cat([pairs_make, morgan_fp_make.unsqueeze(0).unsqueeze(0).expand(pairs_make.size(0), pairs_make.size(1), -1)], dim=2)
        pairs_break = self.pre_reducer_break(pairs_break)
        pairs_make = self.pre_reducer_make(pairs_make)
        pairs_break = self.reducer_break(pairs_break)
        pairs_make = self.reducer_make(pairs_make)
        
        pairs_break = pairs_break.view(-1,MAX_ATOM,MAX_ATOM)
        pairs_make = pairs_make.view(-1,MAX_ATOM,MAX_ATOM)
        
        # Calculate quadruplet probabilities
        quadruplet_probabilities = self.calculate_quadruplet_probabilities(pairs_break, pairs_make)
        
        return pairs_break, pairs_make, quadruplet_probabilities
        
    def calculate_quadruplet_probabilities(self, pairs_break, pairs_make):
        """
        Calculates quadruplet interaction probabilities from break/make logits.

        Ensures symmetry and combines probabilities assuming independence.

        Args:
            pairs_break (Tensor): Logits for breaking edges [MAX_ATOM, MAX_ATOM].
            pairs_make (Tensor): Logits for making edges [MAX_ATOM, MAX_ATOM].

        Returns:
            Tensor: Sigmoid probabilities for quadruplet interactions
                    [MAX_ATOM, MAX_ATOM, MAX_ATOM, MAX_ATOM].
        """
        pairs_break = pairs_break.squeeze(0)
        pairs_make = pairs_make.squeeze(0)
        
        pairs_break = (pairs_break + pairs_break.T) / 2
        pairs_make = (pairs_make + pairs_make.T) / 2

        # Combine the probabilities of destroy and create
        quadruplet_probabilities = pairs_break.unsqueeze(2).unsqueeze(3) + pairs_break.unsqueeze(0).unsqueeze(1) + pairs_make.unsqueeze(1).unsqueeze(3) + pairs_make.unsqueeze(0).unsqueeze(2)
        
        quadruplet_probabilities = torch.sigmoid(quadruplet_probabilities)
        
        return quadruplet_probabilities
 
class GINETimePredictor(torch.nn.Module):
    """
    Predicts a time-related scalar value for a graph using an Enhanced GINE
    architecture.

    Processes node features, edge features (including DOSD), global graph features,
    and uses global mean pooling to obtain a graph-level representation for
    the final prediction.

    Inputs in `forward` (via `data` object):
        - `data.x`: Node features [num_nodes, num_node_features]
        - `data.edge_index`: Graph connectivity [2, num_edges]
        - `data.edge_attr`: Edge features [num_edges, num_edge_features]
        - `data.xA`: Global graph features [num_graphs, num_global_features]
        - `data.distances`: Pairwise distances (unused in forward)
        - `data.dosd_distances`: DOSD distances between nodes [MAX_ATOM, MAX_ATOM]
        - `data.batch`: Batch assignment vector [num_nodes]

    Outputs in `forward`:
        - `x_out`: Predicted time value (scalar, sigmoid scaled to [0, 0.5]) [batch_size, 1]
    """
    def __init__(self):
        super().__init__()
        nn1 = Sequential(Linear(NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv1 = EnhancedGINEConv(nn1, edge_dim=18, global_feat_dim= 21)
        self.ff1 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)
        nn2 = Sequential(Linear(2*NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv2 = EnhancedGINEConv(nn2, edge_dim=18, global_feat_dim= 21)
        self.ff2 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)
        nn3 = Sequential(Linear(2*NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv3 = EnhancedGINEConv(nn3, edge_dim=18, global_feat_dim= 21)
        self.ff3 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)
        
        self.fc3 = nn.Linear(2*NNFEAT, 1)


        
    def forward(self, data):
        x, edge_index, edge_attr, xA, distances, dosd_distances = data.x, data.edge_index, data.edge_attr, data.xA, data.distances, data.dosd_distances

        batch = data.batch

        # Add DOSD distances to edge attributes
        dosd_values = dosd_distances[edge_index[0], edge_index[1]].unsqueeze(1)
        edge_attr_added = torch.cat([edge_attr, dosd_values], dim=1)
        
        # GATN part
        x = F.relu(self.conv1(x, edge_index, xA, edge_attr_added))
        x = self.ff1(x)
        x = F.relu(x)
        
        x = F.relu(self.conv2(x, edge_index, xA, edge_attr_added))
        x = self.ff2(x)
        x = F.relu(x)
        
        x = F.relu(self.conv3(x, edge_index, xA, edge_attr_added))
        x_base = self.ff3(x)
        x_gen = global_mean_pool(x_base, batch)
        
        x_out = torch.sigmoid(self.fc3(x_gen)) * 0.5

        return x_out

class GINETimePredictor_MorganFP(torch.nn.Module):
    """
    Predicts a time-related scalar value for a graph using an Enhanced GINE
    architecture, incorporating Morgan fingerprints.

    Similar to `GINTimePredictor`, but concatenates processed Morgan
    fingerprints to the graph-level representation before the final prediction layer.

    Inputs in `forward` (via `data` object):
        - All inputs from `GINTimePredictor`
        - `data.morgan_fp`: Morgan fingerprints for the graph [num_graphs, morgan_fp_dim]

    Outputs in `forward`:
        - `x_out`: Predicted time value (scalar, sigmoid scaled to [0, 0.5]) [batch_size, 1]
    """
    def __init__(self):
        super().__init__()
        nn1 = Sequential(Linear(NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv1 = EnhancedGINEConv(nn1, edge_dim=18, global_feat_dim= 21)
        self.ff1 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)
        nn2 = Sequential(Linear(2*NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv2 = EnhancedGINEConv(nn2, edge_dim=18, global_feat_dim= 21)
        self.ff2 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)
        nn3 = Sequential(Linear(2*NNFEAT, 2*NNFEAT), ReLU(), Linear(2*NNFEAT, 2*NNFEAT*2*NHEAD))
        self.conv3 = EnhancedGINEConv(nn3, edge_dim=18, global_feat_dim= 21)
        self.ff3 = Linear(2*NNFEAT*2*NHEAD, 2*NNFEAT)

        self.morgan_fp_mlp1 = nn.Sequential(
            Linear(2048, 512),
            nn.ReLU(),
            Linear(512, 256),
            nn.ReLU(),
            Linear(256, 256)
        )

        self.concatenate_morgan_fp = nn.Sequential(
            Linear(2*NNFEAT+256, 2*NNFEAT),
        )
        
        self.fc3 = nn.Linear(2*NNFEAT, 1)

        
    def forward(self, data):
        x, edge_index, edge_attr, xA, distances, dosd_distances, morgan_fp = data.x, data.edge_index, data.edge_attr, data.xA, data.distances, data.dosd_distances, data.morgan_fp

        batch = data.batch

        # Add DOSD distances to edge attributes
        dosd_values = dosd_distances[edge_index[0], edge_index[1]].unsqueeze(1)
        edge_attr_added = torch.cat([edge_attr, dosd_values], dim=1)
        
        # GATN part
        x = F.relu(self.conv1(x, edge_index, xA, edge_attr_added))
        x = self.ff1(x)
        x = F.relu(x)
        
        x = F.relu(self.conv2(x, edge_index, xA, edge_attr_added))
        x = self.ff2(x)
        x = F.relu(x)
        
        x = F.relu(self.conv3(x, edge_index, xA, edge_attr_added))
        x_base = self.ff3(x)
        x_gen = global_mean_pool(x_base, batch)

        morgan_fp_break = self.morgan_fp_mlp1(morgan_fp.float())
        morgan_fp_break = morgan_fp_break.view(-1, 256)
        

        x_in = torch.cat([x_gen, morgan_fp_break], dim=1)

        x_in = self.concatenate_morgan_fp(x_in)
        
        x_out = torch.sigmoid(self.fc3(x_in)) * 0.5

        return x_out

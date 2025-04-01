from lib_functions.libraries import *
from lib_functions.config import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINEConv, global_add_pool, global_max_pool, global_mean_pool

from typing import Callable, Optional, Union
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch import Tensor

class EnhancedGINEConv(GINEConv):
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

class EnhancedGINEConvFPS(GINEConv):
    def __init__(self, nn: torch.nn.Module, global_feat_dim: int, fingerprint_dim: int, eps: float = 0., train_eps: bool = False, edge_dim: Optional[int] = None, **kwargs):
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
            
            # Transform global features and fingerprints to match node features' dimensionality
            self.global_features_transform = Linear(global_feat_dim, in_channels)
            self.fingerprint_transform = Linear(fingerprint_dim, in_channels)  # New transformation for fingerprints
    
    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        global_features: Tensor,  # Global graph features
        fingerprints: Tensor,     # New argument for fingerprints
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # Transform global features
        transformed_global_features = self.global_features_transform(global_features)
        transformed_global_features = transformed_global_features.repeat(x[1].size(0), 1)


        # Transform fingerprints
        transformed_fingerprints = self.fingerprint_transform(fingerprints)
        transformed_fingerprints = transformed_fingerprints.repeat(x[1].size(0), 1)

        # Combine the node features with the global features and fingerprints
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r + transformed_global_features + transformed_fingerprints

        return self.nn(out)

class GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2(torch.nn.Module):
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
        # print(dosd_values)
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

        # Formar todos los pares posibles de embeddings
        x1 = x_base.unsqueeze(1).repeat(1, x_base.size(0), 1)  # Tamaño: [35, 35, 60]
        x2 = x_base.unsqueeze(0).repeat(x_base.size(0), 1, 1)  # Tamaño: [35, 35, 60]

        expanded_xann = xann.unsqueeze(0).expand(x1.size(0), x1.size(1), -1)    
        expanded_x_gen = x_gen.unsqueeze(0).expand(x1.size(0), x1.size(1), -1)

        # Initialize the zero matrix for edge attributes
        # start = datetime.now()
        edge_attr_matrix = torch.zeros(MAX_ATOM, MAX_ATOM, edge_attr.size(1), device=x.device)

        # print("edge_attr")
        # print(edge_attr)
        # Assign edge attributes using advanced indexing
        edge_attr_matrix[edge_index[0], edge_index[1]] = edge_attr
        
        
        # final = datetime.now()
        # print(f"tiempo {final-start}")
        # Concatenate edge attributes to pairs
        pairs = torch.cat([x1, x2, expanded_xann, distances, edge_attr_matrix, dosd_distances.unsqueeze(-1)], dim=2)  # Adjust dimension as per actual sizes

        # every pair of nodes is expanded with graph embedding
        # pairs = torch.cat([x1, x2, expanded_xann, distances], dim=2)  # Tamaño: [N, N, 142]

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
        num_nodes = MAX_ATOM  # Assuming a graph with 35 nodes
        
        pairs_break = pairs_break.squeeze(0)
        pairs_make = pairs_make.squeeze(0)
        
#         # Make the probability matrices symmetric
        pairs_break = (pairs_break + pairs_break.T) / 2
        pairs_make = (pairs_make + pairs_make.T) / 2

#         # Calcular las probabilidades de destruir aristas para todas las combinaciones de quadrupletas
#         prob_destruir = pairs_break.unsqueeze(2).unsqueeze(3) + pairs_break.unsqueeze(0).unsqueeze(1)

#         # Calcular las probabilidades de crear aristas para todas las combinaciones de quadrupletas
#         # Necesitamos reorganizar las dimensiones para alinear correctamente las aristas que se crearán
#         prob_crear = - pairs_make.unsqueeze(1).unsqueeze(3) - pairs_make.unsqueeze(0).unsqueeze(2)

        # Combinar las probabilidades de destruir y crear
        quadruplet_probabilities = pairs_break.unsqueeze(2).unsqueeze(3) + pairs_break.unsqueeze(0).unsqueeze(1) + pairs_make.unsqueeze(1).unsqueeze(3) + pairs_make.unsqueeze(0).unsqueeze(2)
        
        # print(quadruplet_probabilities.shape)

        # Summing over the batch dimension if necessary
#         quadruplet_probabilities = quadruplet_probabilities.sum(dim=0)
        
        quadruplet_probabilities = torch.sigmoid(quadruplet_probabilities)
        
        return quadruplet_probabilities

class GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2_morgan_finetune_2(torch.nn.Module):
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
        # print(dosd_values)
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

        # Formar todos los pares posibles de embeddings
        x1 = x_base.unsqueeze(1).repeat(1, x_base.size(0), 1)  # Tamaño: [35, 35, 60]
        x2 = x_base.unsqueeze(0).repeat(x_base.size(0), 1, 1)  # Tamaño: [35, 35, 60]

        expanded_xann = xann.unsqueeze(0).expand(x1.size(0), x1.size(1), -1)    
        expanded_x_gen = x_gen.unsqueeze(0).expand(x1.size(0), x1.size(1), -1)

        # Initialize the zero matrix for edge attributes
        # start = datetime.now()
        edge_attr_matrix = torch.zeros(MAX_ATOM, MAX_ATOM, edge_attr.size(1), device=x.device)

        # print("edge_attr")
        # print(edge_attr)
        # Assign edge attributes using advanced indexing
        edge_attr_matrix[edge_index[0], edge_index[1]] = edge_attr
        
        
        # final = datetime.now()
        # print(f"tiempo {final-start}")
        # Concatenate edge attributes to pairs
        pairs = torch.cat([x1, x2, expanded_xann, distances, edge_attr_matrix, dosd_distances.unsqueeze(-1)], dim=2)  # Adjust dimension as per actual sizes

        # every pair of nodes is expanded with graph embedding
        # pairs = torch.cat([x1, x2, expanded_xann, distances], dim=2)  # Tamaño: [N, N, 142]

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
        num_nodes = MAX_ATOM  # Assuming a graph with 35 nodes
        
        pairs_break = pairs_break.squeeze(0)
        pairs_make = pairs_make.squeeze(0)
        
#         # Make the probability matrices symmetric
        pairs_break = (pairs_break + pairs_break.T) / 2
        pairs_make = (pairs_make + pairs_make.T) / 2

#         # Calcular las probabilidades de destruir aristas para todas las combinaciones de quadrupletas
#         prob_destruir = pairs_break.unsqueeze(2).unsqueeze(3) + pairs_break.unsqueeze(0).unsqueeze(1)

#         # Calcular las probabilidades de crear aristas para todas las combinaciones de quadrupletas
#         # Necesitamos reorganizar las dimensiones para alinear correctamente las aristas que se crearán
#         prob_crear = - pairs_make.unsqueeze(1).unsqueeze(3) - pairs_make.unsqueeze(0).unsqueeze(2)

        # Combinar las probabilidades de destruir y crear
        quadruplet_probabilities = pairs_break.unsqueeze(2).unsqueeze(3) + pairs_break.unsqueeze(0).unsqueeze(1) + pairs_make.unsqueeze(1).unsqueeze(3) + pairs_make.unsqueeze(0).unsqueeze(2)
        
        # print(quadruplet_probabilities.shape)

        # Summing over the batch dimension if necessary
#         quadruplet_probabilities = quadruplet_probabilities.sum(dim=0)
        
        quadruplet_probabilities = torch.sigmoid(quadruplet_probabilities)
        
        return quadruplet_probabilities
 
class TimePredictionModel_graph(torch.nn.Module):
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

        # # Additional layers to process the concatenated embeddings
        # self.fc_concat1 = Linear(2*NNFEAT + NGFEAT + 167, 256)  # You can adjust the size of 256 if needed
        # self.fc_concat2 = Linear(256, 128)
        # self.fc_output = nn.Linear(128, 1)
        
    def forward(self, data):
        x, edge_index, edge_attr, xA, distances, dosd_distances = data.x, data.edge_index, data.edge_attr, data.xA, data.distances, data.dosd_distances

        batch = data.batch

        # Add DOSD distances to edge attributes
        dosd_values = dosd_distances[edge_index[0], edge_index[1]].unsqueeze(1)
        # print(dosd_values)
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

        # # Concatenate graph embedding with xA and maccs_fp
        # # print(x_gen.shape, xA.shape, maccs_fp.shape)
        # x_combined = torch.cat([x_gen, xA.unsqueeze(0), maccs_fp.unsqueeze(0)], dim=1)  # Concatenate along the feature dimension
        
        # # Pass through additional layers to get the final prediction
        # x_combined = F.relu(self.fc_concat1(x_combined))
        # x_combined = F.relu(self.fc_concat2(x_combined))
        # x_out = torch.sigmoid(self.fc_output(x_combined)) * 0.5
        return x_out

class TimePredictionModel_graph_fps_finetune(torch.nn.Module):
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

        # # Additional layers to process the concatenated embeddings
        # self.fc_concat1 = Linear(2*NNFEAT + NGFEAT + 167, 256)  # You can adjust the size of 256 if needed
        # self.fc_concat2 = Linear(256, 128)
        # self.fc_output = nn.Linear(128, 1)
        
    def forward(self, data):
        x, edge_index, edge_attr, xA, distances, dosd_distances, morgan_fp = data.x, data.edge_index, data.edge_attr, data.xA, data.distances, data.dosd_distances, data.morgan_fp

        batch = data.batch

        # Add DOSD distances to edge attributes
        dosd_values = dosd_distances[edge_index[0], edge_index[1]].unsqueeze(1)
        # print(dosd_values)
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

        # # Concatenate graph embedding with xA and maccs_fp
        # # print(x_gen.shape, xA.shape, maccs_fp.shape)
        # x_combined = torch.cat([x_gen, xA.unsqueeze(0), maccs_fp.unsqueeze(0)], dim=1)  # Concatenate along the feature dimension
        
        # # Pass through additional layers to get the final prediction
        # x_combined = F.relu(self.fc_concat1(x_combined))
        # x_combined = F.relu(self.fc_concat2(x_combined))
        # x_out = torch.sigmoid(self.fc_output(x_combined)) * 0.5
        return x_out

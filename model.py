import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MultiheadAttention, LayerNorm
from concurrent.futures import ThreadPoolExecutor

from data_utils import compute_shortest_path_distances
from  loss import evaluate_unsatisfied_percentage       

class DistanceEncoder(nn.Module):
   
    #Encodes graph distances into embeddings and projects them into attention biases.
    def __init__(self, max_distance=10, embed_dim=8):
        super().__init__()
        self.max_distance = max_distance
        vocab_size = max_distance + 2  # Include an extra bin for "far" distances
        self.embedding = nn.Embedding(vocab_size, embed_dim) # Distance embeddings
        self.proj = nn.Linear(embed_dim, 1) # Projection to scalar bias

    def forward(self, dist):
        # Clamp distances to [0..max_distance + 1] for safe embedding
        dist_clamped = torch.clamp(dist, 0, self.max_distance+1)
        emb = self.embedding(dist_clamped)      # Embed distances(N, N, embed_dim)
        bias = self.proj(emb).squeeze(-1)      #  Project to scalar bias(N, N)
        return bias

class DistanceAwareTransformerLayer(nn.Module):
    #    Transformer layer with distance-aware attention and pre-normalization.
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim), 
            nn.ReLU(),
            nn.Linear(2*embed_dim, embed_dim)
        )
        self.norm1 = LayerNorm(embed_dim) # Pre-norm before attention
        self.norm2 = LayerNorm(embed_dim)  # Pre-norm before feedforward
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # Apply attention with skip connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)  # Add skip connection

        # Apply feedforward network with skip connection
        x_ff = self.norm2(x)
        ff_out = self.ff(x_ff)
        x = x + self.dropout(ff_out)    # skip

        return x

class DistAwareColoringTransformer(nn.Module):
    #  Transformer model for graph coloring with distance-aware attention.
    def __init__(self, embed_dim, num_colors, num_heads=4, n_layers=2, max_dist=10, local_dist=2, distance_encoder=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_colors = num_colors
        self.local_dist = local_dist
        self.n_layers = n_layers

        # Use custom distance encoder if provided, otherwise create a new one
        self.dist_encoder = distance_encoder or DistanceEncoder(max_distance=max_dist, embed_dim=8)
        
        # Stack multiple transformer layers   
        self.layers = nn.ModuleList([
            DistanceAwareTransformerLayer(embed_dim, num_heads) for _ in range(n_layers)
        ])
        self.color_out = nn.Linear(embed_dim, num_colors)

    def forward(self, n_nodes, edges):
        device = next(self.parameters()).device
        
        # 1) Compute BFS distances
        D_list = compute_shortest_path_distances(n_nodes, edges)
        D_tensor = torch.tensor(D_list, dtype=torch.long, device=device)

        # 2) Build float attention mask from distance
        dist_bias = self.dist_encoder(D_tensor)  # shape (N, N)
        INF = 1e9
        attn_mask = torch.where(
            D_tensor <= self.local_dist,
            dist_bias,
            torch.tensor(-INF, device=device)
        )

        # 3) Random node embeddings
        x = torch.randn(1, n_nodes, self.embed_dim, device=device)

        # 4) Pass through each skip-connected Transformer layer
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # 5) Project to color probabilities
        logits = self.color_out(x)               # shape (1, N, num_colors)
        color_probs = F.softmax(logits, dim=-1)  # shape (1, N, num_colors)
        return color_probs.squeeze(0)            # (N, num_colors)

def single_pass(model, num_nodes, edges):
    #Perform one inference pass and calculate unsatisfied constraints.
    color_probs = model(num_nodes, edges)
    unsat_pct = evaluate_unsatisfied_percentage(color_probs, edges)
    return color_probs, unsat_pct

# def multiple_inference_passes(model, num_nodes, edges, passes=5):
#     # Run multiple inference passes with random initialization.
#     # Return the best result (lowest unsatisfied percentage).
#     best_probs = None
#     best_unsat = float('inf')
#     for _ in range(passes):
#         color_probs = model(num_nodes, edges)  # forward pass => random init
#         unsat_pct = evaluate_unsatisfied_percentage(color_probs, edges)
#         if unsat_pct < best_unsat:
#             best_unsat = unsat_pct
#             best_probs = color_probs
#     return best_probs, best_unsat

def multiple_inference_passes_parallel(model, num_nodes, edges, passes=5):
    #Perform parallel inference passes to speed up evaluations.
    best_probs = None
    best_unsat = float('inf')
    with ThreadPoolExecutor(max_workers=passes) as executor:
        futures = [executor.submit(single_pass, model, num_nodes, edges) for _ in range(passes)]
        for future in futures:
            color_probs, unsat_pct = future.result()
            if unsat_pct < best_unsat:
                best_unsat = unsat_pct
                best_probs = color_probs
                if best_unsat == 0:
                    break
    return best_probs, best_unsat
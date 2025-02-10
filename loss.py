import torch 

def coloring_loss(color_probs, edges):
    """
    Computes the negative log-likelihood loss to encourage endpoints of each edge to have different colors.

    - color_probs (torch.Tensor): A tensor of shape (N, C) where:
        - N: Number of nodes in the graph.
        - C: Number of possible colors.
        - Each row is a probability distribution over C colors for a specific node.
    - edges (list of tuples): A list of edges, where each edge is represented as a tuple (u, v).

    - total_loss (float): The average negative log-likelihood loss over all edges.
    """
    device = color_probs.device
    num_colors = color_probs.shape[1]
    A_neq = torch.ones(num_colors, num_colors, device=device)
    A_neq.fill_diagonal_(0)

    eps = 1e-12
    total = 0.
    for (u, v) in edges:
        p_u = color_probs[u]
        p_v = color_probs[v]
        p_edge = torch.einsum('i,ij,j->', p_u, A_neq, p_v)
        total += -torch.log(p_edge + eps)
    return total / len(edges) if edges else 0.0

def evaluate_unsatisfied_percentage(color_probs, edges):
    """
    Calculates the percentage of edges whose endpoints have the same color (conflicts).
    
    - color_probs (torch.Tensor): A tensor of shape (N, C) where:
        - N: Number of nodes in the graph.
        - C: Number of possible colors.
        - Each row is a probability distribution over C colors for a specific node.
    - edges (list of tuples): A list of edges, where each edge is represented as a tuple (u, v).
    """
    assignment = color_probs.argmax(dim=1)  # shape (num_nodes,)
    if not edges:
        return 0.0
    conflicts = sum(assignment[u].item() == assignment[v].item() for (u, v) in edges)
    return 100.0 * conflicts / len(edges)
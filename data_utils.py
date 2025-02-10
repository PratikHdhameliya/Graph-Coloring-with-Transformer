import networkx as nx
import os




def parse_dimacs(filepath):
    """
    Parses a .dimacs file describing an undirected graph.

    Input:
    - filepath (str): Path to the .dimacs file.

    Output:
    - num_nodes (int): Number of nodes in the graph.
    - edges (list of tuples): List of edges in the graph, where each edge is represented as a tuple (u, v).
    """
    edges = []
    num_nodes = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            parts = line.split()
            if parts[0] == 'p':
                num_nodes = int(parts[2])
            elif parts[0] == 'e':
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                edges.append((u, v))
    return num_nodes, edges

def load_dimacs_graphs(folder):
    """
    Loads all .dimacs files in the given folder.

    Input:
    - folder (str): Path to the folder containing .dimacs files.

    Output:
    - graphs (list of tuples): List of (num_nodes, edges) for each graph.
      - num_nodes (int): Number of nodes in the graph.
      - edges (list of tuples): List of edges in the graph, where each edge is represented as a tuple (u, v).
    """
    paths = [os.path.join(folder, fname) for fname in os.listdir(folder) if fname.endswith('.dimacs')]
    graphs = []
    for p in paths:
        n, e = parse_dimacs(p)
        graphs.append((n, e))
    return graphs


def compute_shortest_path_distances(n_nodes, edges):
    """
    Computes the shortest-path distances between all pairs of nodes in an undirected, unweighted graph.

    Input:
    - n_nodes (int): Number of nodes in the graph.
    - edges (list of tuples): List of edges in the graph, where each edge is represented as a tuple (u, v).

    Output:
    - D (list of lists): 2D list (n_nodes x n_nodes) where D[u][v] is the shortest-path distance between node u and node v.
      If nodes u and v are disconnected, D[u][v] is set to (n_nodes + 1).
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges)

    INF = n_nodes + 1
    D = [[INF]*n_nodes for _ in range(n_nodes)]

    for start in range(n_nodes):
        D[start][start] = 0
        queue = [start]
        visited = {start}
        dist_so_far = 0
        while queue:
            next_queue = []
            for u in queue:
                for v in G[u]:
                    if v not in visited:
                        visited.add(v)
                        D[start][v] = D[start][u] + 1
                        next_queue.append(v)
            queue = next_queue

    return D

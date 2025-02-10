# Graph Coloring with Transformer

This repository demonstrates a novel approach to solving the **graph coloring problem** using a **distance-aware transformer model**. The implementation utilizes **PyTorch** and various utility libraries to implement and train the transformer model on a dataset of DIMACS-format graphs.

---

## Problem Statement

Graph coloring is the task of assigning a color to each node in a graph such that no two connected nodes share the same color. It is a classic problem in combinatorial optimization and has various applications in:

- Scheduling problems  
- Register allocation in compilers  
- Frequency assignment in wireless networks  

Here, we aim to minimize the **percentage of unsatisfied edges** (i.e., edges where both endpoints share the same color) using a transformer-based approach.

---

## Methodology

### Data Format

The dataset consists of graphs in the **DIMACS format**, where each file specifies:  
- The number of nodes and edges.  
- The edges connecting pairs of nodes.  

The graph dataset is zipped and must be extracted before running the code.

### Model

We use a **Distance-Aware Transformer Model** with the following components:

1. **Distance Encoder**:  
   - Encodes pairwise distances between nodes into a scalar bias, which influences attention computations.

2. **Transformer Layers**:  
   - Layers equipped with skip connections and pre-normalization for stability.  
   - Includes multi-head attention and feed-forward networks to learn relationships between nodes.

3. **Loss Function**:  
   - Negative log-likelihood of assigning different colors to connected nodes is used as the loss function.

---

### Approach

- Graph distances are computed using BFS for each node.  
- A distance-based attention mask is applied to focus on local interactions.  
- The transformer outputs probabilities for each node's color assignment.  

### **Visualization of the Graph Coloring Solution**
Below, you can see visualizations of our model's output on dense and power-law graphs. The nodes are colored based on the model’s predictions, aiming to minimize conflicts (unsatisfied edges).  

**Dense Graph Coloring:**  
![Image](https://github.com/user-attachments/assets/58327f5e-2452-4e74-a7e3-cf69c17e2d1c)

**Power-Law Graph Coloring:**  
![Image](https://github.com/user-attachments/assets/cd6fd81e-76d2-4cf0-af4a-7580beb4f4ce)

---

## Installation

Before running the code, ensure all required packages are installed. Use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

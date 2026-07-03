import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def spring_layout(vals, seed=42):
    n = len(vals)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    np.random.seed(seed)
    pos0 = {i: np.array([float(vals[i]), np.random.random()]) for i in range(n)}

    pos_spring = nx.spring_layout(
        G,
        pos=pos0,
        dim=2,
        k=0.01,
        iterations=10000,
        seed=seed
    )

    # 5) Overwrite x‐coords to their original values (lock x, let only y move)
    for i in range(n):
        pos_spring[i][0] = pos0[i][0]
    y = np.array([pos_spring[i][1] for i in range(n)])
    return y
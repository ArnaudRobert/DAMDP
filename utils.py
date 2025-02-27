import networkx as nx


def get_layers(G):
    """Given a Directed Acyclic Graph
       return a dictionary containing the
       topological layers of the graph"""
    layers = {}
    node2layers = nx.get_node_attributes(G, 'layer')
    for k, v in node2layers.items():
        if v in layers:
            layers[v].append(k)
        else:
            layers[v] = [k]
    return layers

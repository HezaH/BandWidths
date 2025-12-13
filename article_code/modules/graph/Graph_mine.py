


class Node:
    def __init__(self, value, label=None):
        self.value = value
        self.label = label
        self.neighbors = []

    def add_neighbor(self, node):
        if node not in self.neighbors:
            self.neighbors.append(node)
            node.add_neighbor(self)  # doubly-linked

    def get_neighbors(self):
        return self.neighbors

    def bandwidth(self):
        return len(self.neighbors)


class Graph:
    def __init__(self, edges=[]):
        self.nodes = {}
        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def add_node(self, value, label=None):
        if value not in self.nodes:
            self.nodes[value] = Node(value, label)

    def add_edge(self, v1, v2):
        if v1 not in self.nodes:
            self.add_node(v1)
        if v2 not in self.nodes:
            self.add_node(v2)
        
        self.nodes[v1].add_neighbor(self.nodes[v2])

    def bandwidth(self):
        return max([node.bandwidth() for _, node in self.nodes.items()])



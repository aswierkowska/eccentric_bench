from qiskit.providers.fake_provider import GenericBackendV2
# from qiskit.visualization import plot_coupling_map
from qiskit.transpiler import CouplingMap
import rustworkx as rx
import math

def generate_cube_map(num_layers, num_rows, num_columns, bidirectional=True):
    """Return a coupling map of qubits connected in a 3D cube structure."""
    def get_index(layer, row, col):
        return layer * (num_rows * num_columns) + row * num_columns + col
    
    graph = rx.PyDiGraph()
    graph.add_nodes_from(range(num_layers * num_rows * num_columns))
    
    for layer in range(num_layers):
        for row in range(num_rows):
            for col in range(num_columns):
                node = get_index(layer, row, col)
                
                if col < num_columns - 1:
                    neighbor = get_index(layer, row, col + 1)
                    graph.add_edge(node, neighbor, None)
                    if bidirectional:
                        graph.add_edge(neighbor, node, None)

                if row < num_rows - 1:
                    neighbor = get_index(layer, row + 1, col)
                    graph.add_edge(node, neighbor, None)
                    if bidirectional:
                        graph.add_edge(neighbor, node, None)
                
                if layer < num_layers - 1:
                    neighbor = get_index(layer + 1, row, col)
                    graph.add_edge(node, neighbor, None)
                    if bidirectional:
                        graph.add_edge(neighbor, node, None)

    return CouplingMap(graph.edge_list(), description="cube")


def get_custom_topology(shape: str, num_qubits: int):
    if shape == "line":
        coupling_map = CouplingMap.from_line(num_qubits)
    elif shape == "grid":
        #num_rows, num_cols = int(num_qubits**(1/2)), int(num_qubits**(1/2))
        num_rows = 15
        num_cols = 20
        coupling_map = CouplingMap.from_grid(num_rows, num_cols)
    elif shape == "cube":
        #num_layers, num_rows, num_cols = int(num_qubits**(1/3)), int(num_qubits**(1/3)), int(num_qubits**(1/3))
        num_layers = 5
        num_rows = 6
        num_cols = 10
        coupling_map = generate_cube_map(num_layers, num_rows, num_cols)
        # TODO: requires graphviz installed, useful for the paper
        # plot_coupling_map(num_qubits, None, coupling_map.get_edges(), filename="graph.png")
    elif shape == "full":
        coupling_map = CouplingMap.from_full(num_qubits)
    elif shape == "heavyhex":
        d = (2 + math.sqrt(40 * num_qubits + 24)) / 10
        d = int(d)
        if d % 2 == 0:
            d -= 1
        coupling_map = CouplingMap.from_heavy_hex(d)
    else:
        return None
    
    return GenericBackendV2(coupling_map.size(), coupling_map=coupling_map)


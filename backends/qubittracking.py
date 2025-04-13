from .utils import get_backend
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import QuantumCircuit, transpile
import rustworkx as rx

class QubitTracking:
    def __init__(self, backend : GenericBackendV2, circuit: QuantumCircuit):
        self.backend = backend
        self.qubit_mapping = {}
        self.physical_qubits = backend.num_qubits
        #for i in range(self.physical_qubits):
        #    self.qubit_mapping[i] = i
        self.qubit_mapping = dict(enumerate(circuit.layout.initial_index_layout()))
        

        self.coupling_map = backend.coupling_map
        self.coupling_map_graph = self.coupling_map.graph.to_undirected()
    
        
    def swap_qubits(self, logical_qubit1: int, logical_qubit2: int):
        physical_qubit1 = self.get_layout_postion(logical_qubit1)
        physical_qubit2 = self.get_layout_postion(logical_qubit2)

        self.qubit_mapping[logical_qubit1] = physical_qubit2
        self.qubit_mapping[logical_qubit2] = physical_qubit1


    def get_neighbours(self,logical_qubit: int):
        physical_qubit = self.get_layout_postion(logical_qubit)
        neighbours = self.coupling_map_graph.neighbors(physical_qubit)
        logical_neighbours = [self.get_logical_qubit(neighbour) for neighbour in neighbours]
        return logical_neighbours
    
    def get_logical_qubit(self, physical_qubit: int):
        """Get logical qubit from physical qubit"""
        for logical_qubit, p_qubit in self.qubit_mapping.items():
            if p_qubit == physical_qubit:
                return logical_qubit
        raise ValueError(f"Physical qubit {physical_qubit} not found in mapping.")

    def get_layout_postion(self, logical_qubit: int):
        """Get physical position of logical qubit"""
        try:
            physical_qubit = self.qubit_mapping[logical_qubit]
            return physical_qubit
        except KeyError:
            raise ValueError(f"Logical qubit {logical_qubit} not found in mapping.")
        
    

if __name__ == "__main__":
    # Example usage
    qc = QuantumCircuit(4)
    qc.h(2)
    qc.cx(0, 1)
    qc.cx(2, 0)
    qc.cx(1, 2)
    qc.cx(1, 3)
    num_qubits = 8
    backend = get_backend("custom_cube", num_qubits)
    qc = transpile(
        qc,
        backend=backend
    )
    qubit_tracker = QubitTracking(backend, qc)
    print(qubit_tracker.qubit_mapping)
    qubit_tracker.swap_qubits(0, 1)
    qubit_tracker.swap_qubits(0, 2)
    print(qubit_tracker.qubit_mapping)
    print(qubit_tracker.get_neighbours(0))
    print(qubit_tracker.get_neighbours(1))
    print(qubit_tracker.get_neighbours(2))
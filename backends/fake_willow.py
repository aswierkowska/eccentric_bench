import numpy as np
from qiskit.transpiler import InstructionProperties, Target, CouplingMap
from qiskit.circuit.library import RZGate, XGate, SXGate, CZGate, Measure, Reset
# TODO: Delay
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2, Options
from qiskit.visualization import plot_coupling_map

class FakeGoogleWillowBackend(BackendV2):
    """Fake Google Willow Backend."""
    
    def __init__(self, extended=False):
        super().__init__(name="FakeGoogleWillow", backend_version=2)
        if extended:
            self._qubit_positions = self._get_extended_qubit_positions()
        else:
            self._qubit_positions = self._get_real_qubit_positions()
        self._coupling_map = CouplingMap(self._get_coupling_list(self._qubit_positions))
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake Google Willow", num_qubits=self._num_qubits)


    @property
    def target(self):
        return self._target
    
    @property
    def max_circuits(self):
        return None
    
    @property
    def coupling_map(self):
        return self._coupling_map
    
    @property
    def qubit_positions(self):
        return self._qubit_positions
    
    @property
    def get_remote_gates(self):
        return []
    
    @property
    def num_qubits(self):
        return self._num_qubits
    
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
    
    @classmethod
    def _get_real_qubit_positions(cls):
        qubit_positions = [
            (0, 6), (0, 7), (0, 8),
            (1, 5), (1, 6), (1, 7), (1, 8),
            (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
            (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
            (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12),
            (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12),
            (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
            (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13),
            (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12),
            (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11),
            (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
            (11, 6), (11, 7), (11, 8), (11, 9),
            (12, 6), (12, 7), (12, 8)
        ]
        max_row = max(row for row, col in qubit_positions)
        return [(max_row - row, col) for row, col in qubit_positions]
    
    @classmethod
    def _get_extended_qubit_positions(cls):
        qubit_positions_1 = [
            (0, 6), (0, 7), (0, 8),
            (1, 5), (1, 6), (1, 7), (1, 8),
            (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
            (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
            (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12),
            (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12),
            (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
            (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13),
            (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12),
            (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11),
            (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
            (11, 6), (11, 7), (11, 8), (11, 9),
            (12, 6), (12, 7), (12, 8)
        ]

        qubit_positions_2 = [(x - 7, y + 7) for x, y in qubit_positions_1]
        qubit_positions_3 = [(x - 14, y + 14) for x, y in qubit_positions_1]

        qubit_positions = qubit_positions_1 + qubit_positions_2 + qubit_positions_3
        max_row = max(row for row, col in qubit_positions)
        return [(max_row - row, col) for row, col in qubit_positions]
    
    @classmethod
    def _get_coupling_list(cls, qubit_positions):
        coupling_list = []
        pos_to_index = {pos: idx for idx, pos in enumerate(qubit_positions)}
        
        for row, col in qubit_positions:
            if (row + 1, col) in pos_to_index:
                coupling_list.append((pos_to_index[(row, col)], pos_to_index[(row + 1, col)]))
            if (row, col + 1) in pos_to_index:
                coupling_list.append((pos_to_index[(row, col)], pos_to_index[(row, col + 1)]))
        return coupling_list
    
    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")

if __name__ == "__main__":
    backend = FakeGoogleWillowBackend()
    backend_ext = FakeGoogleWillowBackend(extended=True)
    plot_coupling_map(backend.coupling_map.size(), backend.qubit_positions, backend.coupling_map.get_edges(), filename="willow.png")
    plot_coupling_map(backend_ext.coupling_map.size(), backend_ext.qubit_positions, backend_ext.coupling_map.get_edges(), filename="willow_ext.png")

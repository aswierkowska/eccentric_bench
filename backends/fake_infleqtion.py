import numpy as np
import re
import random
import stim
from qiskit.transpiler import InstructionProperties, Target, CouplingMap
from qiskit.circuit.library import RZGate, XGate, SXGate, CZGate, Measure, Reset
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2, Options
from qiskit.visualization import plot_coupling_map
from qiskit.providers import QubitProperties

# Taken from https://arxiv.org/pdf/2408.08288

class FakeInfleqtionBackend(BackendV2):
    """Fake Infleqtion Backend."""
    def __init__(self, extended=False):
        super().__init__(name="FakeInfleqtion", backend_version=2)
        if extended:
            self.rows = 16
            self.columns = 24
        else:
            # rescaled 4x each side
            self.rows = 4
            self.columns = 6
        self._coupling_map = CouplingMap.from_grid(self.rows, self.columns)
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake Infleqtion NA", num_qubits=self._num_qubits)
        self.addStateOfTheArtQubits()
        self._remote_gates = self.add_shuttling_connections()

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
        return self._remote_gates
    
    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def coupling_map(self):
        return self._coupling_map
    
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
    
    def addStateOfTheArtQubits(self):
        qubit_props = []

        t1 = 26 # atom loss lifetime
        t2 = 2.8 # 2.8 seconds estimated T2 with decoupling (12.7(5) ms without)

        for i in range(self.num_qubits):
            qubit_props.append(QubitProperties(t1=t1, t2=t2))

        self.target.qubit_properties = qubit_props

    def add_shuttling_connections(self):
        existing_edges = set(self._coupling_map.get_edges())
        all_pairs = {(q1, q2) for q1 in range(self.num_qubits) for q2 in range(q1 + 1, self.num_qubits)}
        new_edges = all_pairs - existing_edges
        for edge in new_edges:
            self._coupling_map.add_edge(edge[0], edge[1])
        return list(new_edges)

    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")

if __name__ == "__main__":
    backend = FakeInfleqtionBackend()
    plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="infleqtion.png")
    print(backend.coupling_map)

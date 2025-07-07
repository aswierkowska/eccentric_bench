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

class FakeInfleqtionNoShuttlingBackend(BackendV2):
    """Fake Infleqtion Backend."""
    def __init__(self, extended=False):
        super().__init__(name="FakeInfleqtion", backend_version=2)
        if extended:
            self.rows = 12
            self.columns = 18
        else:
            # rescaled 3x each side
            self.rows = 4
            self.columns = 6
        self._coupling_map = CouplingMap.from_grid(self.rows, self.columns)
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake Infleqtion No Shuttling NA", num_qubits=self._num_qubits)
        self.addStateOfTheArtQubits()
        self._remote_gates = {}

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

    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")

if __name__ == "__main__":
    backend = FakeInfleqtionNoShuttlingBackend()
    plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="infleqtion.png")
    print(backend.coupling_map)
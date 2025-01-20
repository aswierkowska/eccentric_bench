import numpy as np
import rustworkx as rx
 
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CZGate, ECRGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_gate_map
 
 
class FakeLargeBackend(BackendV2):
    """Fake multi chip backend."""
 
    def __init__(self, distance=11, number_of_chips=1):
        """Instantiate a new fake multi chip backend.
 
        Args:
            distance (int): The heavy hex code distance to use for each chips'
                coupling map. This number **must** be odd. The distance relates
                to the number of qubits by:
                :math:`n = \\frac{5d^2 - 2d - 1}{2}` where :math:`n` is the
                number of qubits and :math:`d` is the ``distance``
            number_of_chips (int): The number of chips to have in the multichip backend
                each chip will be a heavy hex graph of ``distance`` code distance.
        """
        super().__init__(name="FakeLargeBackend", backend_version=2)
        # Create a heavy-hex graph using the rustworkx library, then instantiate a new target
        self._graph = rx.generators.directed_heavy_hex_graph(
            distance, bidirectional=False
        )
        num_qubits = len(self._graph) * number_of_chips
        self._target = Target(
            "Fake multi-chip backend", num_qubits=num_qubits
        )

        self.version = 2
 
        # Generate instruction properties for single qubit gates and a measurement, delay,
        #  and reset operation to every qubit in the backend.
        rng = np.random.default_rng(seed=12345678942)
        rz_props = {}
        x_props = {}
        sx_props = {}
        measure_props = {}
        delay_props = {}
 
        # Add 1q gates. Globally use virtual rz, x, sx, and measure
        for i in range(num_qubits):
            qarg = (i,)
            rz_props[qarg] = InstructionProperties(error=0.0, duration=0.0)
            x_props[qarg] = InstructionProperties(
                error=rng.uniform(1e-6, 1e-4),
                duration=rng.uniform(1e-8, 9e-7),
            )
            sx_props[qarg] = InstructionProperties(
                error=rng.uniform(1e-6, 1e-4),
                duration=rng.uniform(1e-8, 9e-7),
            )
            measure_props[qarg] = InstructionProperties(
                error=rng.uniform(1e-3, 1e-1),
                duration=rng.uniform(1e-8, 9e-7),
            )
            delay_props[qarg] = None
        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(SXGate(), sx_props)
        self._target.add_instruction(RZGate(Parameter("theta")), rz_props)
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(Reset(), measure_props)
 
        self._target.add_instruction(Delay(Parameter("t")), delay_props)
        # Add chip local 2q gate which is CZ
        cz_props = {}
        for i in range(number_of_chips):
            for root_edge in self._graph.edge_list():
                offset = i * len(self._graph)
                edge = (root_edge[0] + offset, root_edge[1] + offset)
                cz_props[edge] = InstructionProperties(
                    error=rng.uniform(7e-4, 5e-3),
                    duration=rng.uniform(1e-8, 9e-7),
                )
        self._target.add_instruction(CZGate(), cz_props)
 
    @property
    def target(self):
        return self._target
 
    @property
    def max_circuits(self):
        return None
 
    @property
    def graph(self):
        return self._graph
 
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
 
    def run(self, circuit, **kwargs):
        raise NotImplementedError(
            "This backend does not contain a run method"
        )

if __name__ == "__main__":
    backend = FakeLargeBackend()
    
    plot_gate_map(backend.graph, plot_directed=False)
import numpy as np
from qiskit.transpiler import InstructionProperties, Target, CouplingMap
from qiskit.circuit.library import RZGate, XGate, SXGate, CZGate, Measure, Reset
# TODO: Delay
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2, Options
from qiskit.visualization import plot_coupling_map

class FakeContinuumApolloBackend(BackendV2):
    """Fake Continnum Apollo Backend."""
    
    def __init__(self):
        super().__init__(name="FakeContinuumApollo", backend_version=2)
        # if 192 probably 12 x 16
        # then 1000s could be 1728 (36*48) or 3072 (48*64)
        self._coupling_map = CouplingMap.from_grid(36, 48)
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake Continuum Apollo", num_qubits=self._num_qubits)
        

        """
        rng = np.random.default_rng(seed=12345678942)
        rz_props, x_props, sx_props, measure_props, delay_props = {}, {}, {}, {}, {}
        
        for i in range(self._num_qubits):
            qarg = (i,)
            rz_props[qarg] = InstructionProperties(error=0.0, duration=0.0)
            x_props[qarg] = InstructionProperties(error=0.00035, duration=rng.uniform(1e-8, 9e-7))
            sx_props[qarg] = InstructionProperties(error=0.00035, duration=rng.uniform(1e-8, 9e-7))
            measure_props[qarg] = InstructionProperties(error=0.0077, duration=rng.uniform(1e-8, 9e-7))
            delay_props[qarg] = None
        
        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(SXGate(), sx_props)
        self._target.add_instruction(RZGate(Parameter("theta")), rz_props)
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(Reset(), measure_props)
        
        cz_props = {}
        for edge in self._coupling_map.get_edges():
            cz_props[edge] = InstructionProperties(error=0.0033, duration=rng.uniform(1e-8, 9e-7))
        self._target.add_instruction(CZGate(), cz_props)
        """

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
    def num_qubits(self):
        return self._num_qubits
    
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
  
    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")

if __name__ == "__main__":
    backend = FakeContinuumApolloBackend()
    plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="apollo.png")
import numpy as np
from qiskit.transpiler import Target, CouplingMap
from qiskit.providers import BackendV2, Options
from qiskit.visualization import plot_coupling_map
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers import QubitProperties
from qiskit_ibm_runtime.fake_provider import FakeKyiv

class GridMCMBackend(BackendV2):
    
    def __init__(self):
        super().__init__(name="FakeMCM", backend_version=2)
        self._coupling_map = CouplingMap.from_grid(20, 20)
        self._remote_gates = []
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake MCM", num_qubits=self._num_qubits) # TODO: hardware limitations
        self.addStateOfTheArtQubits()
        self.gate_set = ["id", "sx", "x", "rz", "rzz", "cz", "rx"]

    @property
    def target(self):
        return self._target
    
    @property
    def max_circuits(self):
        return None
    
    @property
    def get_remote_gates(self):
        return self._remote_gates
    
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

    
    def addStateOfTheArtQubits(self):
        np.random.seed(123)
        qubit_props = []
        
        for i in range(self._num_qubits):
            #t1 = np.random.normal(190, 120, 1)
            #t1 = np.clip(t1, 50, 500)
            t1 = 142.02 * 1e-6

            #t2 = np.random.normal(130, 120, 1)
            #t2 = np.clip(t2, 50, 650)
            t2 = 100.95 * 1e-6

            qubit_props.append(QubitProperties(t1=t1, t2=t2, frequency=5.0e9))

        self.target.qubit_properties = qubit_props



    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")


if __name__ == "__main__":
    backend = FakeIBMFlamingo()
    print(backend.get_remote_gates)
    #plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="flamingo.png")

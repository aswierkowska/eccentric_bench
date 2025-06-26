import numpy as np
from qiskit.transpiler import Target, CouplingMap
from qiskit.providers import BackendV2, Options
from qiskit.providers import QubitProperties

class VarianceBackend(BackendV2):
    """Full for Testing Variance Backend."""
    
    def __init__(self, variance: str):
        super().__init__(name="VarianceBackend", backend_version=2)
        # Based on Flamingo
        self._remote_gates, self._coupling_map = [], CouplingMap.from_full(399)
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake IBM Flamingo", num_qubits=self._num_qubits)
        if variance == "low":
            self.addStateOfTheArtQubits(60)
        elif variance == "mid":
            self.addStateOfTheArtQubits(120)
        elif variance == "high":
            self.addStateOfTheArtQubits(180)
        

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
    def get_remote_gates(self):
        return []

    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def get_remote_gates(self):
        return []
    
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
     
    def addStateOfTheArtQubits(self, variance=120):
        qubit_props = []
       
        for i in range(self._num_qubits):
            t1 = np.random.normal(190, variance, 1)
            t1 = np.clip(t1, 0, 570)
            t1 = t1 * 1e-6
            
            t2 = np.random.normal(130, variance, 1)
            t2 = np.clip(t2, 0, 390)
            t2 = t2 * 1e-6
        
            qubit_props.append(QubitProperties(t1=t1, t2=t2, frequency=5.0e9))
        
        self.target.qubit_properties = qubit_props



    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")


if __name__ == "__main__":
    backend = VarianceBackend("mid")
    print(f"Backend Name: {backend.name}")
    print(f"Number of Qubits: {backend.num_qubits}")
    for i, qubit in enumerate(backend.target.qubit_properties[:5]):
        if not (0 <= qubit.t1[0] <= 570e-6) or not (0 <= qubit.t2[0] <= 390e-6):
            print(f"Qubit {i} INCORRECT: T1 = {qubit.t1[0]:.2e}, T2 = {qubit.t2[0]:.2e}")

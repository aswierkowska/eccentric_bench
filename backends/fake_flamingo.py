import numpy as np
from qiskit.transpiler import InstructionProperties, Target, CouplingMap
from qiskit.circuit.library import RZGate, XGate, SXGate, CZGate, Measure, Reset
# TODO: Delay
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2, Options
from qiskit.visualization import plot_coupling_map
from qiskit_ibm_runtime import QiskitRuntimeService

class FakeIBMFlamingo(BackendV2):
    """Fake IBM Flamingo Backend."""
    
    def __init__(self):
        super().__init__(name="FakeIBMFlamingo", backend_version=2)
        #backend_large = customBackend(name="FezDQC_",num_qubits=399, coupling_map=map_large, )
        self._remote_gates, self._coupling_map = self.get_endpoints()
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake IBM Flamingo", num_qubits=self._num_qubits) # TODO: hardware limitations

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
    

    def heavySquareHeronCouplingMap(self):
        service = QiskitRuntimeService(instance="ibm-q/open/main")
        backend = service.backend("ibm_kyiv")

        base_coupling_map = backend.coupling_map

        base_coupling_map.add_edge(13, 127)
        base_coupling_map.add_edge(113, 128) 
        base_coupling_map.add_edge(117, 129)
        base_coupling_map.add_edge(121, 130)
        base_coupling_map.add_edge(124, 131)
        base_coupling_map.add_edge(128, 132)

        return base_coupling_map

 
    def DQCCouplingMap(self, coupling_map1: CouplingMap, coupling_map2: CouplingMap, endpoints: list):
        edges_1 = coupling_map1.get_edges()
        last_qubit = coupling_map1.size()
        converted_edges_1 = [[i, j] for i,j in edges_1]
        new_endpoints = [[i, last_qubit + j] for i,j in endpoints]
        new_coupling_map2 = [[last_qubit + i, last_qubit + j] for i,j in coupling_map2]
        
        new_coupling_map = CouplingMap(converted_edges_1 + new_endpoints + new_coupling_map2)
        new_endpoints_tuple = [(i, j) for i,j in new_endpoints]

        return new_endpoints_tuple, new_coupling_map

    def get_endpoints(self):
        endpoints, map_interm = self.DQCCouplingMap(self.heavySquareHeronCouplingMap(), self.heavySquareHeronCouplingMap(), [[32, 18], [51, 37], [70, 56], [89, 75]])
        endpoints_new, map_large = self.DQCCouplingMap(map_interm, self.heavySquareHeronCouplingMap(), [[165, 18], [184, 37], [203, 56], [222, 75]])
        all_endpoints = endpoints + endpoints_new
        return all_endpoints, map_large

    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")


if __name__ == "__main__":
    backend = FakeIBMFlamingo()
    print(backend.get_remote_gates)
    #plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="flamingo.png")
from qiskit.transpiler import Target, CouplingMap
from qiskit.providers import BackendV2, Options, QubitProperties
from qiskit.visualization import plot_coupling_map
import numpy as np


class FakeIBMLoon(BackendV2):
    """Fake IBM Loon Backend."""

    def __init__(self):
        super().__init__(name="FakeIBMLoon", backend_version="2")

        # self._coupling_map = self.get_coupling_map()
        self._remote_gates, self._coupling_map = self.get_distributed_coupling_map()
        self._num_qubits = self._coupling_map.size()

        # TODO: hardware limitations
        self._target = Target("Fake IBM Loon", num_qubits=self._num_qubits)
        
        self.addStateOfTheArtQubits()
        self.gate_set = ["id", "sx", "x", "rz", "rzz", "cz", "rx"]

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
    def get_remote_gates(self):
        return self._remote_gates
    
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

    def squareLatticeCouplingMap(self, add_c_couplers: bool = False):
        """ Construct coupling map of a square lattice for the Nighthawk processor 
        
        The coupling map consists of a simple square lattice connectivity, with additional special c_couplers
        connecting distant nodes.

        Information taken from:
        - https://www.ibm.com/quantum/blog/large-scale-ftqc

        TODO: Add variable amount of c_couplers. Currently only the first row can have distant connections.
        """
        
        # Nighthawk specific configuration for 120 qubits
        rows = 11
        cols = 11

        # Vertical and horizontal distance of c_couplers
        c_coupler_distance = 6

        def idx(r, c):
            """ Indexing function. """
            return r * cols + c

        # Start at the bottom left and iteratively connect to right and upper neighbour
        edges = []
        for r in range(rows):
            for c in range(cols):
                q = idx(r, c)

                # Connect right neighbour
                if c < cols - 1:
                    edges.append((q, idx(r, c + 1)))

                # Connect upper neighbour
                if r < rows - 1:
                    edges.append((q, idx(r + 1, c)))

                if add_c_couplers:
                    # C-coupler connections
                    if r + c_coupler_distance < rows and c + c_coupler_distance/2 < cols and r == 0:
                        edges.append((q, idx(r + c_coupler_distance, c + int(c_coupler_distance/2))))
                    
                    if r + c_coupler_distance/2 < rows and c + c_coupler_distance < cols and r == 0:
                        edges.append((q, idx(r + int(c_coupler_distance/2), c + c_coupler_distance)))

        return CouplingMap(edges)
    
    def DQCCouplingMap(self, coupling_map1: CouplingMap, coupling_map2: CouplingMap, endpoints: list):
        edges_1 = coupling_map1.get_edges()
        last_qubit = coupling_map1.size()
        converted_edges_1 = [[i, j] for i,j in edges_1]
        new_endpoints = [[i, last_qubit + j] for i,j in endpoints]
        new_coupling_map2 = [[last_qubit + i, last_qubit + j] for i,j in coupling_map2]
        
        new_coupling_map = CouplingMap(converted_edges_1 + new_endpoints + new_coupling_map2)
        new_endpoints_tuple = [(i, j) for i,j in new_endpoints]

        return new_endpoints_tuple, new_coupling_map

    def get_coupling_map(self) -> CouplingMap:
        return self.squareLatticeCouplingMap(add_c_couplers=True)

    def get_distributed_coupling_map(self) -> tuple[list, CouplingMap]:
        """ Create coupling map of nighthawk architecture by combining multiple loon chips.
        
        Similar to fake_flamingo, multiple coupling maps are connected at specific locations and are combined into a
        single coupling map. The connections are returned as endpoints.

        TODO: Check if the specified endpoints make sense 
        """

        endpoints, coupling_map_two = self.DQCCouplingMap(self.squareLatticeCouplingMap(add_c_couplers=True),
                                                    self.squareLatticeCouplingMap(add_c_couplers=True),
                                                    [[32, 18], [51, 37], [70, 56], [89, 75]]
                                                    )

        return endpoints, coupling_map_two

    def addStateOfTheArtQubits(self):
        qubit_props = []
        
        for i in range(self._num_qubits):
            t1 = np.random.normal(190, 120, 1)
            t1 = np.clip(t1, 50, 500)
            t1 = t1 * 1e-6

            t2 = np.random.normal(130, 120, 1)
            t2 = np.clip(t2, 50, 650)
            t2 = t2 * 1e-6

            qubit_props.append(QubitProperties(t1=t1, t2=t2, frequency=5.0e9))

        self.target.qubit_properties = qubit_props
    

if __name__ == "__main__":
    backend = FakeIBMLoon()

    plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="loon.png", planar=False)

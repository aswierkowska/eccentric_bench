from qiskit_ibm_runtime import QiskitRuntimeService
from .custom_topologies import get_custom_topology
from .fake_apollo import FakeQuantinuumApolloBackend
from .fake_flamingo import FakeIBMFlamingo
from .fake_willow import FakeGoogleWillowBackend
from .fake_infleqtion import FakeInfleqtionBackend
from .fake_infleqtion_no_shuttling import FakeInfleqtionNoShuttlingBackend
from .variance_backend import VarianceBackend

def get_backend(backend: str, backend_size: int):
    backend_type = backend.split("_")[0]
    if backend_type == "ibm":
        service = QiskitRuntimeService(instance="ibm-q/open/main")
        backend = service.backend(backend)
    elif backend_type == "custom" and backend_size:
        shape = backend.split("_")[1]
        backend = get_custom_topology(shape, backend_size)
    elif backend_type == "real":
        name = backend.split("_")[1]
        if name == "flamingo":
            backend = FakeIBMFlamingo()
        elif name == "willow":
            # By default we allow extension as otherwise codes are limited
            backend = FakeGoogleWillowBackend(extended=True)
        elif name == "apollo":
            backend = FakeQuantinuumApolloBackend()
        elif name == "infleqtion":
            backend = FakeInfleqtionBackend(extended=True)
        elif name == "nsinfleqtion":
            backend = FakeInfleqtionNoShuttlingBackend(extended=True)
        #elif name == "aquila":
        #    backend = FakeQueraAquilaBackend()
        else:
            raise NotImplementedError
    elif backend_type == "variance":
        name = backend.split("_")[1]
        backend = VarianceBackend(name)
    else:
        raise NotImplementedError
    return backend

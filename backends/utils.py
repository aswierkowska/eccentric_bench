from qiskit_ibm_runtime import QiskitRuntimeService
from .custom_topologies import get_custom_topology

def get_backend(backend: str, backend_size: int):
    backend_type = backend.split("_")[0]
    if backend_type == "ibm":
        service = QiskitRuntimeService(instance="ibm-q/open/main")
        backend = service.backend(backend)
    elif backend_type == "custom" and backend_size:
        shape = backend.split("_")[1]
        backend = get_custom_topology(shape, backend_size)
    else:
        raise NotImplementedError
    return backend
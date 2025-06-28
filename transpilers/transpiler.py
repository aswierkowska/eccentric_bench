from qiskit import transpile
from .translators import qiskit_stim

def run_transpiler(circuit, backend, layout_method, routing_method):
    return transpile(
                circuit,
                basis_gates=qiskit_stim,
                optimization_level=0,
                backend=backend,
                layout_method=layout_method,
                routing_method=routing_method
        )

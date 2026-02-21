from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
        SabreLayout,
        BasisTranslator
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from .translators import qiskit_stim_gates

def run_transpiler(circuit, backend, layout_method, routing_method):
    #pm = PassManager([
    #    SabreLayout(backend.coupling_map),
    #    BasisTranslator(SessionEquivalenceLibrary,
    #                target_basis=qiskit_stim_gates)
    #])

    #mapped = pm.run(circuit)
    #return mapped
    return transpile(
                circuit,
                basis_gates=qiskit_stim_gates,
                optimization_level=0,
                backend=backend,
                layout_method= layout_method,
                routing_method=routing_method
    )

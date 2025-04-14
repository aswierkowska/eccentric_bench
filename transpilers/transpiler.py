from qiskit import transpile
from qiskit.transpiler import PassManager
from .shuttling_routing import ShuttlingRouting
from .translators import qiskit_stim_gates

def run_transpiler(circuit, backend_name, backend, layout_method, routing_method):
    if backend_name in ["real_aquila", "real_apollo"]:
        print("IN CORRECT 1")
        pass_manager = PassManager()
        pass_manager.append([ShuttlingRouting(backend.coupling_map)])
        print("IN CORRECT 2")
        qc = transpile(
            circuit,
            basis_gates=qiskit_stim_gates,
            optimization_level=0,
            backend=backend,
            layout_method=layout_method,
            routing_method=None
        )
        print("IN CORRECT 3")
        return pass_manager.run(qc)
    else:
        return transpile(
                circuit,
                basis_gates=qiskit_stim_gates,
                optimization_level=0,
                backend=backend,
                layout_method=layout_method,
                routing_method=routing_method
        )
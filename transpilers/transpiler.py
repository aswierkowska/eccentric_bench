from qiskit import transpile
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from .shuttling_routing import ShuttlingRouting
from .translators import qiskit_stim_gates

def run_transpiler(circuit, backend_name, backend, layout_method, routing_method):
    if backend_name in ["real_aquila", "real_apollo"]:
        shuttling_routing_pass = ShuttlingRouting(backend.coupling_map)
        base_pm = generate_preset_pass_manager(
            backend=backend,
            basis_gates=qiskit_stim_gates,
            optimization_level=0,
            layout_method=layout_method,
            routing_method="none"
        )
        base_pm.routing = PassManager([shuttling_routing_pass])
        print("IN CORRECT 2")
        return base_pm.run(circuit)
    else:
        return transpile(
                circuit,
                basis_gates=qiskit_stim_gates,
                optimization_level=0,
                backend=backend,
                layout_method=layout_method,
                routing_method=routing_method
        )
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit import Gate
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.layout import Layout

class ShuttlingSwap(Gate):
    def __init__(self):
        super().__init__('shuttling_swap', 2, [])

    def _define(self):
        q = QuantumRegister(2, 'q')
        definition = QuantumCircuit(q)
        definition.append(SwapGate(), [q[0], q[1]])
        self.definition = definition

##############################################################################################################################################
# Based on: https://github.com/Qiskit/qiskit/blob/13443f4c975756ded21345f8837c3d1e526c33ef/qiskit/transpiler/passes/routing/basic_swap.py
##############################################################################################################################################
class ShuttlingRouting(TransformationPass):
    def __init__(self, coupling_map, fake_run=False):
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self.fake_run = fake_run

    def run(self, dag):
        if self.fake_run:
            return self._fake_run(dag)

        new_dag = dag.copy_empty_like()

        if self.coupling_map is None:
            raise TranspilerError("BasicSwap cannot run with coupling_map=None")

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Basic swap runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")
        print("DISJOINT")
        disjoint_utils.require_layout_isolated_to_component(
            dag, self.coupling_map if self.target is None else self.target
        )
        print("AFTER")
        canonical_register = dag.qregs["q"]
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        for layer in dag.serial_layers():
            subdag = layer["graph"]

            print(f"Layer {layer} processing started.")
            for gate in subdag.two_qubit_ops():
                print(f"Processing gate {gate}")
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]
                # MODIFIED CODE TO INCLUDE SHUTTLING SWAP
                dist = self.coupling_map.distance(physical_q0, physical_q1)
                print(f"Distance between {physical_q0} and {physical_q1}: {dist}")
                if dist > 1:
                    swap_gate = SwapGate() if dist == 2 else ShuttlingSwap()
                    print(f"Applying {swap_gate} for qubits {physical_q0}, {physical_q1}")
                    swap_layer = DAGCircuit()
                    swap_layer.add_qreg(canonical_register)
                    swap_layer.apply_operation_back(swap_gate, (physical_q0, physical_q1), cargs=(), check=False)
                    
                    order = current_layout.reorder_bits(new_dag.qubits)
                    new_dag.compose(swap_layer, qubits=order)
                    current_layout.swap(physical_q0, physical_q1)
                        
            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)
            if new_dag.depth() > 2000:
                raise RuntimeError("Aborting: DAG depth exploded — possible infinite swap loop")

        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = current_layout
        else:
            self.property_set["final_layout"] = current_layout.compose(
                self.property_set["final_layout"], dag.qubits
            )

        return new_dag
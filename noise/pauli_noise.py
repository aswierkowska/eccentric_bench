from qiskit_qec.noise import PauliNoiseModel
from qiskit import QuantumCircuit

def generate_pauli_error(p: float) -> PauliNoiseModel:
    pnm = PauliNoiseModel()
    pnm.add_operation(
        "h", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p}
    )
    pnm.add_operation("cx", {"ix": p / 3, "xi": p / 3, "xx": p / 3, "ii": 1 - p})
    pnm.add_operation("id", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p})
    pnm.add_operation("measure", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p})
    return pnm


def generate_circuit_specific_pauli_error(circuit: QuantumCircuit, p: float) -> PauliNoiseModel:
    gates = set([operation.name for operation in circuit.data])
    pnm = PauliNoiseModel()
    for gate in gates:
        if gate in ["cx", "cy", "cz", "swap"]:
            pnm.add_operation(
                gate, {"ix": p / 3, "xi": p / 3, "xx": p / 3, "ii": 1 - p}
            )
        elif gate in ["x", "y", "z", "h", "s", "s_dag", "reset", "measure"]:
            pnm.add_operation(gate, {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p})
    return pnm
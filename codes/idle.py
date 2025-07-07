import stim
from qiskit_qec.circuits import StimCodeCircuit

def make_idle_qubit_circuit(num_rounds: int, num_qubits: int) -> StimCodeCircuit:
    circuit = stim.Circuit()

    for q in range(num_qubits):
        circuit.append("QUBIT_COORDS", [q, 0])

    for _ in range(num_rounds):
        circuit.append("TICK")

    # Measure all qubits at the end in Z basis
    circuit.append("M", list(range(num_qubits)))

    return StimCodeCircuit(stim_circuit=circuit)

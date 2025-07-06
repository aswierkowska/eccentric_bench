import stim
from qiskit_qec.circuits import StimCodeCircuit

def make_idle_qubit_circuit(num_rounds: int) -> stim.Circuit:
    circuit = stim.Circuit()
    circuit.append("QUBIT_COORDS", [0, 0])
    for _ in range(num_rounds):
        circuit.append("TICK")

    circuit.append("M", [0])

    return StimCodeCircuit(stim_circuit = circuit)
import qiskit
import stim


def count_total_gates_qiskit(circuit: qiskit.QuantumCircuit):
    """
    Count the number of gates in a Qiskit circuit.
    """
    count = 0
    gates = circuit.count_ops()
    for gate, num_gates in gates.items():
        if gate not in ["measure", "reset", "barrier"]:
            count += num_gates
    return count

def count_swaps_qiskit(circuit: qiskit.QuantumCircuit):
    gates = circuit.count_ops()
    return gates["swap"]
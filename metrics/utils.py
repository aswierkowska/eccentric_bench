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
    if "swap" in gates:
        return gates["swap"]
    return 0

def count_total_gates_stim(circuit: stim.Circuit):
    """
    Count the number of gates in a Stim circuit.
    """
    count = 0
    for op in circuit.operations():
        if op.name not in ["MEASURE", "RESET", "BARRIER"]:
            count += 1
    return count

def stabilizer_fidelity():
    return None

def get_threshold_error_rate():
    return None

def get_resource_overhead():
    return None

def get_error_supression_factor():
    return None


if __name__ == "__main__":
    # Example usage for stim
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=3,
        distance=5
    )
    print("Total gates in stim circuit:", count_total_gates_stim(stim_circuit))

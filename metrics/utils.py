import qiskit
import stim

import os


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
    #write circuit to file
    with open("circuit.txt", "w") as f:
        f.write(str(circuit))
    #read circuit from file
    with open("circuit.txt", "r") as f:
        circuit_str = f.readlines()

    # Count the number of gates
    count = 0
    repeat_cycles = 0
    repeating = False
    for line in circuit_str:
        line = line.strip()
        if line.startswith("REPEAT"):
            #seperate the line by space
            l = line.split(" ")
            repeat_cycles = int(l[1])
            repeating = True
        if line.startswith("}"):
            repeating = False
            repeat_cycles = 0
        if line.startswith("H ") or line.startswith("S ") or line.startswith("T ") or line.startswith("X ") or line.startswith("Y ") or line.startswith("Z "):
            #seperate the line by space
            l = line.split(" ")
            count += int(len(l)-1) * max(1, repeat_cycles*repeating)
            
        elif line.startswith("CX ") or line.startswith("CZ "):
            #seperate the line by space
            l = line.split(" ")
            count += int((int((len(l)-1)) * max(1, repeat_cycles*repeating))/2)
            
    os.remove("circuit.txt")     
    
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
        rounds=4,
        distance=3
    )
    print("Total gates in stim circuit:", count_total_gates_stim(stim_circuit))

import qiskit
import stim
import stim_gate_list

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

def detailed_gate_count_qiskit(circuit: qiskit.QuantumCircuit):
    """Returns a dictionary with the number of gates in a Qiskit circuit."""
    return circuit.count_ops()

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
        if line.startswith("H ") or line.startswith("S ") or line.startswith("X ") or line.startswith("Y ") or line.startswith("Z "):
            #seperate the line by space
            l = line.split(" ")
            count += int(len(l)-1) * max(1, repeat_cycles*repeating)
            
        elif line.startswith("CX ") or line.startswith("CZ "):
            #seperate the line by space
            l = line.split(" ")
            count += int((int((len(l)-1)) * max(1, repeat_cycles*repeating))/2)

    os.remove("circuit.txt")     
    
    return count

def detailed_gate_count_stim(circuit: stim.Circuit):
    gates = {}
    with open("circuit.txt", "w") as f:
        f.write(str(circuit))
    #read circuit from file
    with open("circuit.txt", "r") as f:
        circuit_str = f.readlines()
    
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

        if line.startswith("H ") or line.startswith("S ") or line.startswith("X ") or line.startswith("Y ") or line.startswith("Z "):
            #seperate the line by space
            l = line.split(" ")
            gate = l[0]
            if gate not in gates:
                gates[gate] = 0
            gates[gate] += int(len(l)-1) * max(1, repeat_cycles*repeating)
        elif line.startswith("CX ") or line.startswith("CZ "):
            #seperate the line by space
            l = line.split(" ")
            gate = l[0]
            if gate not in gates:
                gates[gate] = 0
            gates[gate] += int((int((len(l)-1)) * max(1, repeat_cycles*repeating))/2)
        else: # uncommon gates
            l = line.split(" ")
            if l[0] in stim_gate_list.SINGLE_QUBIT_GATES:
                if l[0] not in gates:
                    gates[l[0]] = 0
                gates[l[0]] += int(len(l)-1) * max(1, repeat_cycles*repeating)
            elif l[0] in stim_gate_list.TWO_QUBIT_GATES:
                if l[0] not in gates:
                    gates[l[0]] = 0
                gates[gate] += int((int((len(l)-1)) * max(1, repeat_cycles*repeating))/2)

    #os.remove("circuit.txt")
    return gates


def tableau_fidelity(t1: stim.Tableau, t2: stim.Tableau) -> float:
    """Function by Craig Gidney
        soruce: https://quantumcomputing.stackexchange.com/questions/38826/how-do-i-efficiently-compute-the-fidelity-between-two-stabilizer-tableau-states
    
    """
    t3 = t2**-1 * t1
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(t3)

    p = 1
    for q in range(len(t3)):
        e = sim.peek_z(q)
        if e == -1:
            return 0
        if e == 0:
            p *= 0.5
            sim.postselect_z(q, desired_value=False)

    return p

def stabilizer_fidelity(starting_circuit, transpiled_circuit):
    #check if stim type otherwise value error for now
    if type(starting_circuit) == qiskit.QuantumCircuit or type(transpiled_circuit) == qiskit.QuantumCircuit:
        raise ValueError("Stabilizer fidelity only works for stim circuits")
    
    tableau_1 = stim.Tableau.from_circuit(starting_circuit, ignore_measurement=True, ignore_reset=True)
    tableau_2 = stim.Tableau.from_circuit(transpiled_circuit, ignore_measurement=True, ignore_reset=True)
    fidelity = tableau_fidelity(tableau_1, tableau_2)
    return fidelity


def get_resource_overhead_total_gates(starting_circuit, transpiled_circuit):
    if type(starting_circuit) == qiskit.QuantumCircuit:
        starting_gates = count_total_gates_qiskit(starting_circuit)
    elif type(starting_circuit) == stim.Circuit:
        starting_gates = count_total_gates_stim(starting_circuit)
    else:
        raise ValueError("Unsupported circuit type")

    if type(transpiled_circuit) == qiskit.QuantumCircuit:
        transpiled_gates = count_total_gates_qiskit(transpiled_circuit)
    elif type(transpiled_circuit) == stim.Circuit:
        transpiled_gates = count_total_gates_stim(transpiled_circuit)
    else:
        raise ValueError("Unsupported circuit type")

    gate_diff = transpiled_gates - starting_gates

    return gate_diff

def get_error_supression_factor_logical(logical_error_rate_low, logical_error_rate_high):
    big_lamda  = logical_error_rate_low/logical_error_rate_high if logical_error_rate_high != 0 else -1
    return big_lamda

def get_error_supression_factor_physical(physical_error_threshold, physical_error_rate):
    big_lamda = physical_error_threshold/physical_error_rate if physical_error_rate != 0 else -1
    return big_lamda



def get_threshold_error_rate():
    #TODO: implement this function  
    return None

if __name__ == "__main__":
    # Example usage for stim
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=3,
        distance=3
    )

    print("detailed gate count stim: ", detailed_gate_count_stim(stim_circuit))
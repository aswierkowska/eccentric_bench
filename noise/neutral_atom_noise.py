import stim
import re

def add_neutral_atom_noise(circuit, p_after_clifford_depolarization=0, p_after_reset_flip_probability=0,
                           p_before_measure_flip_probability=0, p_before_round_data_depolarization=0):

    p_after_clifford_depolarization = float(p_after_clifford_depolarization)
    p_after_reset_flip_probability = float(p_after_reset_flip_probability)
    p_before_measure_flip_probability = float(p_before_measure_flip_probability)
    p_before_round_data_depolarization = float(p_before_round_data_depolarization)

    noisy_circuit = stim.Circuit()
    round_number = 0 # Track the error correction round
    round_type = "encoding" # track if encoding or repeating round.

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitInstruction):
            command = instruction.name
            targets = instruction.targets_copy()

            if command == "R" or command == "RX":
                noisy_circuit.append(command, targets)
                noisy_circuit.append("X_ERROR" if command == "R" else "Z_ERROR", targets, p_after_reset_flip_probability)

            elif command == "CNOT":
                noisy_circuit.append(command, targets)
                noisy_circuit.append("DEPOLARIZE2", targets, p_after_clifford_depolarization)

            elif command == "H":
                noisy_circuit.append(command, targets)
                noisy_circuit.append("DEPOLARIZE1", targets, p_after_clifford_depolarization)

            elif command == "TICK":
                noisy_circuit.append(command)
                round_number += 1
                if round_number > 8:
                    round_number = 1
                    if round_type == "encoding":
                        round_type = "repeating"
                    else:
                        round_type = "encoding"

            elif command == "MR" or command == "MX" or command == "M":
                noisy_circuit.append("X_ERROR" if command == "MR" or command == "M" else "Z_ERROR", targets, p_before_measure_flip_probability)
                noisy_circuit.append(command, targets)

            elif command == "DEPOLARIZE1":
                noisy_circuit.append(command, targets, p_before_round_data_depolarization)

            elif command == "DEPOLARIZE2":
                pass

            elif command == "X_ERROR" or command == "Z_ERROR":
                pass

            elif command == "DETECTOR":
                noisy_circuit.append(command, targets)

            elif command == "OBSERVABLE_INCLUDE":
                instruction_str = str(instruction)
                match = re.search(r"OBSERVABLE_INCLUDE\((\d+)\)", instruction_str)
                if match:
                    observable_index = int(match.group(1))
                    noisy_circuit.append(command, targets, observable_index)
                else:
                    raise ValueError("Could not parse observable index from OBSERVABLE_INCLUDE instruction.")

            else:
                noisy_circuit.append(command, targets)

            #Add data qubit depolarizing noise.
            if command == "TICK" and round_type == "repeating":
                for i in range(circuit.num_qubits // 2, circuit.num_qubits):
                    noisy_circuit.append("DEPOLARIZE1", i, p_before_round_data_depolarization)

        elif isinstance(instruction, stim.CircuitRepeatBlock):
            noisy_block = add_neutral_atom_noise(instruction.body_copy(),
                                                p_after_clifford_depolarization,
                                                p_after_reset_flip_probability,
                                                p_before_measure_flip_probability,
                                                p_before_round_data_depolarization)
            for _ in range(instruction.repeat_count):
                noisy_circuit += noisy_block
        else:
            raise TypeError(f"Unexpected instruction type: {type(instruction)}")

    return noisy_circuit
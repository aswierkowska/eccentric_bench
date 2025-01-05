from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import stim

def apply_gates_and_operations(stim_circuit, qiskit_circuit, qreg, creg):
    qubit_coords = {}  # Holds the current coordinates of qubits
    detector_markers = []  # Store detector markers (if needed)
    observable_markers = []  # Store observable markers (if needed)
    measurement_index = 0

    for instruction in stim_circuit:
        if isinstance(instruction, stim.CircuitInstruction):
            gate_name = instruction.name
            targets = instruction.targets_copy()
            args = instruction.gate_args_copy()

            if gate_name == "QUBIT_COORDS":
                # Store the coordinates of the qubit
                qubit_index = int(targets[0].value)
                coordinates = list(args)
                qubit_coords[qubit_index] = coordinates

            elif gate_name == "RX":
                # Apply RX (reset and measure in X-basis)
                for target in targets:
                    qubit_index = target.value
                    qiskit_circuit.reset(qreg[qubit_index])
                    qiskit_circuit.h(qreg[qubit_index])  # Prepare qubit in X-basis
                    if creg:
                        qiskit_circuit.measure(qreg[qubit_index], creg[measurement_index])
                        measurement_index += 1

            elif gate_name == "R":
                # Handle R gate as reset operation
                for target in targets:
                    qubit_index = target.value
                    if creg:
                        # Measure the qubit
                        qiskit_circuit.measure(qreg[qubit_index], creg[measurement_index])
                        measurement_index += 1
                    # Reset the qubit
                    qiskit_circuit.reset(qreg[qubit_index])

            elif gate_name == "MR":
                # Measure and reset
                for target in targets:
                    qubit_index = target.value
                    if creg:
                        qiskit_circuit.measure(qreg[qubit_index], creg[measurement_index])
                        measurement_index += 1
                    qiskit_circuit.reset(qreg[qubit_index])

            elif gate_name == "M":
                # Measure in the default basis
                for target in targets:
                    qubit_index = target.value
                    if creg:
                        qiskit_circuit.measure(qreg[qubit_index], creg[measurement_index])
                        measurement_index += 1

            elif gate_name == "MX":
                # Measure in X-basis
                for target in targets:
                    qubit_index = target.value
                    qiskit_circuit.h(qreg[qubit_index])  # Transform to X-basis
                    if creg:
                        qiskit_circuit.measure(qreg[qubit_index], creg[measurement_index])
                        measurement_index += 1
                    qiskit_circuit.h(qreg[qubit_index])  # Restore original basis

            elif gate_name in ["H", "X", "Y", "Z", "S", "S_DAG", "T", "T_DAG"]:
                # Apply single qubit gates
                for target in targets:
                    qubit_index = target.value
                    getattr(qiskit_circuit, gate_name.lower())(qreg[qubit_index])

            elif gate_name in ["CX", "SWAP", "CY", "CZ"]:
                # Apply two qubit gates
                qubit_indices = [target.value for target in targets]
                if len(qubit_indices) == 2:
                    getattr(qiskit_circuit, gate_name.lower())(
                        qreg[qubit_indices[0]], qreg[qubit_indices[1]]
                    )

            elif gate_name == "DETECTOR":
                # Store detector markers for post-processing (if needed)
                detector_markers.append([target.value for target in targets])

            elif gate_name == "OBSERVABLE_INCLUDE":
                # Store observable markers for post-processing (if needed)
                observable_markers.append([target.value for target in targets])

            elif gate_name == "SHIFT_COORDS":
                # Handle SHIFT_COORDS by updating qubit coordinates
                shift_x, shift_y, _ = args
                for qubit, coords in qubit_coords.items():
                    qubit_coords[qubit] = [coords[0] + shift_x, coords[1] + shift_y]

            elif gate_name == "TICK":
                # In Stim, TICK is a no-op. In Qiskit, we use a barrier.
                ##qiskit_circuit.barrier()
                continue

            else:
                raise ValueError(f"Unexpected operation in Stim circuit: {gate_name}")

        elif isinstance(instruction, stim.CircuitRepeatBlock):
            # Handle repeat block by applying the subcircuit repeatedly
            repeat_count = instruction.repeat_count
            subcircuit = instruction.body_copy()

            for _ in range(repeat_count):
                apply_gates_and_operations(subcircuit, qiskit_circuit, qreg, creg)



def get_qiskit_circuits(stim_circuit: "stim.Circuit"):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    num_qubits = stim_circuit.num_qubits
    num_clbits = stim_circuit.num_measurements

    # Create Qiskit registers
    qreg = QuantumRegister(num_qubits, "q")
    creg = ClassicalRegister(num_clbits, "c") if num_clbits > 0 else None
    qiskit_circuit = QuantumCircuit(qreg, creg) if creg else QuantumCircuit(qreg)

    # Iterate through Stim circuit instructions and apply gates
    apply_gates_and_operations(stim_circuit, qiskit_circuit, qreg, creg)

    return qiskit_circuit

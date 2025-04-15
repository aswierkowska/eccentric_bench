import numpy as np
import re
import random
import stim
from qiskit.transpiler import InstructionProperties, Target, CouplingMap
from qiskit.circuit.library import RZGate, XGate, SXGate, CZGate, Measure, Reset
# TODO: Delay
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2, Options
from qiskit.visualization import plot_coupling_map
from qiskit.providers import QubitProperties

# TODO: can be taken from Braket

class FakeQueraAquilaBackend(BackendV2):
    """Fake QuEra Aquila Backend."""
    
    def __init__(self):
        super().__init__(name="FakeQueraAquila", backend_version=2)
        # coupling map can be fairly arbitrary
        # TODO: atom sorting probability?
        self._coupling_map = CouplingMap.from_grid(16, 16)
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake QuEra Aquila", num_qubits=self._num_qubits)
        # Neutral-atom-specific noise parameters
        self.p_atom_loss = 0.005  # Atoms can be lost over time
        self.p_detection_error = 0.05  # Detection errors in mid-circuit measurement
        self.p_rabi_noise = 0.02  # Imperfections in laser pulses (over/under-rotation)
        self.p_detuning_noise = 0.02  # Variations in laser frequency (Z-error)
        self.p_crosstalk = 0.02  # Nearby qubit interference
        self.p_swap_noisy = 0.02  # Noisy logical SWAP (via CNOTs)
        self.p_swap_low_noise = 0.001  # Low-noise SWAP (physical qubit rearrangement)
        self.p_after_clifford_depolarization = 0.005  # Gate depolarization
        self.p_after_reset_flip_probability = 0.01  # Reset errors
        self.p_before_measure_flip_probability = 0.02 # Readout errors
        self.p_before_round_data_depolarization = 0.002  # Memory decoherence

    @property
    def target(self):
        return self._target
    
    @property
    def max_circuits(self):
        return None
    
    @property
    def coupling_map(self):
        return self._coupling_map
    
    @property
    def qubit_positions(self):
        return self._qubit_positions
    
    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def coupling_map(self):
        return self._coupling_map
    
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
    
    def addStateOfTheArtQubits(self):
        qubit_props = []
        
        for i in range(self.num_qubits):
            qubit_props.append(QubitProperties(t1=100, t2=50))

        self.target.qubit_properties = qubit_props


    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")
    
    # adding device-specific noise to stim circuit
    def add_realistic_noise(self, circuit):
        """Adds realistic neutral-atom noise to a stim circuit."""
        noisy_circuit = stim.Circuit()
        round_number = 0
        round_type = "encoding"

        for instruction in circuit:
            if isinstance(instruction, stim.CircuitInstruction):
                command = instruction.name
                targets = instruction.targets_copy()

                if command in {"R", "RX"}:
                    noisy_circuit.append(command, targets)
                    noisy_circuit.append("X_ERROR" if command == "R" else "Z_ERROR", targets, self.p_after_reset_flip_probability)

                elif command == "CNOT":
                    noisy_circuit.append(command, targets)
                    noisy_circuit.append("DEPOLARIZE2", targets, self.p_after_clifford_depolarization)

                elif command == "H":
                    noisy_circuit.append(command, targets)
                    noisy_circuit.append("DEPOLARIZE1", targets, self.p_after_clifford_depolarization)

                elif command == "SWAP":
                    print(targets)
                    for i, q in enumerate(targets):
                        if i % 2 == 1:
                            q1, q2 = targets[i-1], q
                            # Check if SWAP is "virtual" (low-noise) or logical (noisy)
                            distance = abs(q1 - q2)

                    #if distance > 5:  # Long-range: Atoms physically moved, low noise
                    #    noisy_circuit.append(command, targets)
                    #    noisy_circuit.append("DEPOLARIZE2", targets, self.p_swap_low_noise)
                    #else:  # Logical SWAP via CNOTs, higher noise
                    #    noisy_circuit.append("CNOT", [q1, q2])
                    #    noisy_circuit.append("CNOT", [q2, q1])
                    #    noisy_circuit.append("CNOT", [q1, q2])
                    #    noisy_circuit.append("DEPOLARIZE2", [q1, q2], self.p_swap_noisy)

                elif command == "TICK":
                    noisy_circuit.append(command)
                    round_number += 1
                    if round_number > 8:
                        round_number = 1
                        round_type = "repeating" if round_type == "encoding" else "encoding"

                elif command in {"MR", "MX", "M"}:
                    noisy_circuit.append("X_ERROR" if command in {"MR", "M"} else "Z_ERROR", targets, self.p_before_measure_flip_probability)
                    noisy_circuit.append(command, targets)

                elif command == "DEPOLARIZE1":
                    noisy_circuit.append(command, targets, self.p_before_round_data_depolarization)

                elif command in {"DEPOLARIZE2", "X_ERROR", "Z_ERROR"}:
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

                ## --- Introduce Dominant Neutral Atom Errors ---
                if command != "TICK":
                    for q in targets:
                        # Simulate **atom loss** (qubit disappears)
                        if random.random() < self.p_atom_loss:
                            noisy_circuit.append("DEPOLARIZE1", [q], 1.0)

                        # Simulate **rydberg blockade detection errors**
                        if random.random() < self.p_detection_error:
                            noisy_circuit.append("X_ERROR", [q], self.p_detection_error)

                        # Simulate **detuning noise (Z-error from laser shifts)**
                        if random.random() < self.p_detuning_noise:
                            noisy_circuit.append("Z_ERROR", [q], self.p_detuning_noise)

                        # Simulate **Rabi frequency noise (random over/under-rotations)**
                        if random.random() < self.p_rabi_noise:
                            noisy_circuit.append("PAULI_CHANNEL_1", [q], [self.p_rabi_noise, self.p_rabi_noise, self.p_rabi_noise])

                    # **Crosstalk errors** (neighboring qubits affecting each other)
                    if len(targets) > 1:
                        noisy_circuit.append("CORRELATED_ERROR", targets, self.p_crosstalk)

                ## --- Periodic Data Qubit Decoherence ---
                if command == "TICK" and round_type == "repeating":
                    for i in range(circuit.num_qubits // 2, circuit.num_qubits):
                        noisy_circuit.append("DEPOLARIZE1", i, self.p_before_round_data_depolarization)

            elif isinstance(instruction, stim.CircuitRepeatBlock):
                noisy_block = self.add_realistic_noise(instruction.body_copy())
                for _ in range(instruction.repeat_count):
                    noisy_circuit += noisy_block

            else:
                raise TypeError(f"Unexpected instruction type: {type(instruction)}")

        return noisy_circuit

if __name__ == "__main__":
    backend = FakeQueraAquilaBackend()
    plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="aquila.png")

from typing import Dict, Tuple, Optional, Set
import stim
import math
from .noise import *
from backends import QubitTracking

apollo_err_prob = {
    "P_CZ": 0,
    "P_CZ_CROSSTALK": 0,
    "P_CZ_LEAKAGE": 0,
    "P_IDLE": 0,
    "P_READOUT": 0,
    "P_RESET": 0,
    "P_SQ": 0,
    "P_LEAKAGE": 0,
    "P_SHUTTLING_SWAP": 0,
}

# TODO: add SHUTTLING_SWAP time
apollo_gate_times = {}

class ApolloNoise(NoiseModel):
    def __init__(
        self,
        idle: float,
        measure_reset_idle: float,
        noisy_gates: Dict[str, float],
        noisy_gates_connection: Dict[str, float],
        qt: QubitTracking,
        any_clifford_1: Optional[float] = None,
        any_clifford_2: Optional[float] = None,
        use_correlated_parity_measurement_errors: bool = False
    ):
        self.idle = idle
        self.measure_reset_idle = measure_reset_idle
        self.noisy_gates = noisy_gates
        self.noisy_gates_connection = noisy_gates_connection
        self.qt = qt
        self.any_clifford_1 = any_clifford_1
        self.any_clifford_2 = any_clifford_2
        self.use_correlated_parity_measurement_errors = use_correlated_parity_measurement_errors

    @staticmethod
    def get_noise(qt: QubitTracking) -> 'ApolloNoise':
        return ApolloNoise(
            idle=apollo_err_prob["P_IDLE"],
            measure_reset_idle=apollo_err_prob["P_RESET"],
            noisy_gates={
                "CX": apollo_err_prob["P_CZ"],
                "CZ": apollo_err_prob["P_CZ"],
                "CZ_CROSSTALK": apollo_err_prob["P_CZ_CROSSTALK"],
                "CZ_LEAKAGE": apollo_err_prob["P_CZ_LEAKAGE"],
                "R": apollo_err_prob["P_RESET"],
                "H": apollo_err_prob["P_SQ"],
                "M": apollo_err_prob["P_READOUT"],
                "MPP": apollo_err_prob["P_READOUT"],
                "SWAP": apollo_err_prob["P_CZ"],
                "SHUTTLING_SWAP": apollo_err_prob["P_SHUTTLING_SWAP"],
            },
            noisy_gates_connection={
                "CX": apollo_err_prob["P_CZ"] + 0.3,
            },
            qt=qt,
            use_correlated_parity_measurement_errors=True
        )

    def update_swaps(self, op: stim.CircuitInstruction):
        targets = op.targets_copy()
        for i in range(0, len(targets), 2):
            q1 = targets[i].qubit_value
            q2 = targets[i+1].qubit_value
            self.qt.swap_qubits(q1, q2)

    def get_qubit_err_prob(self, qubit: int, gate_duration_ns: float) -> float:
        qubit_properties = self.backend.qubit_properties(qubit)
        t1 = qubit_properties.t1
        t2 = qubit_properties.t2
        t = gate_duration_ns
        p_relax = 1 - math.exp(-t / t1)
        p_dephase = 1 - math.exp(-t / t2)
        return (p_relax + p_dephase - p_relax * p_dephase)

    def noisy_op(self, op: stim.CircuitInstruction, base_p: float, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()

        if op.name in ANY_CLIFFORD_1_OPS:
            for t in targets:
                q = t.value
                qubit_p = self.get_qubit_err_prob(q, apollo_gate_times[op.name])
                combined_p = 1 - (1 - base_p) * (1 - qubit_p)
                if combined_p > 0:
                    post.append_operation("DEPOLARIZE1", [q], combined_p)
                mid.append_operation(op.name, [t], args)

        elif op.name in ANY_CLIFFORD_2_OPS or op.name in SWAP_OPS:
            for i in range(0, len(targets), 2):
                q1 = targets[i].value
                q2 = targets[i+1].value
                p1 = self.get_qubit_err_prob(q1, apollo_gate_times[op.name])
                p2 = self.get_qubit_err_prob(q2, apollo_gate_times[op.name])
                combined_p1 = 1 - (1 - base_p) * (1 - p1)
                combined_p2 = 1 - (1 - base_p) * (1 - p2)
                if combined_p1 > 0:
                    post.append_operation("DEPOLARIZE1", [q1], combined_p1)
                if combined_p2 > 0:
                    post.append_operation("DEPOLARIZE1", [q2], combined_p2)
                mid.append_operation(op.name, [targets[i], targets[i+1]], args)

        elif op.name in RESET_OPS or op.name in MEASURE_OPS:
            for t in targets:
                q = t.value
                qubit_p = self.get_qubit_err_prob(q, apollo_gate_times[op.name])
                combined_p = 1 - (1 - base_p) * (1 - qubit_p)
                if combined_p > 0:
                    if op.name in RESET_OPS:
                        post.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", [q], combined_p)
                    if op.name in MEASURE_OPS:
                        pre.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", [q], combined_p)
                mid.append_operation(op.name, [t], args)

    def noisy_circuit(self, circuit: stim.Circuit, *, qs: Optional[Set[int]] = None) -> stim.Circuit:
        result = stim.Circuit()
        ancilla = circuit.num_qubits

        current_moment_pre = stim.Circuit()
        current_moment_mid = stim.Circuit()
        current_moment_post = stim.Circuit()
        used_qubits: Set[int] = set()
        measured_or_reset_qubits: Set[int] = set()
        if qs is None:
            qs = set(range(circuit.num_qubits))

        def flush():
            nonlocal result
            if not current_moment_mid:
                return
            idle_qubits = sorted(qs - used_qubits)
            if used_qubits and idle_qubits and self.idle > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.idle)
            idle_qubits = sorted(qs - measured_or_reset_qubits)
            if measured_or_reset_qubits and idle_qubits and self.measure_reset_idle > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.measure_reset_idle)

            result += current_moment_pre
            result += current_moment_mid
            result += current_moment_post
            used_qubits.clear()
            current_moment_pre.clear()
            current_moment_mid.clear()
            current_moment_post.clear()
            measured_or_reset_qubits.clear()

        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                flush()
                result += self.noisy_circuit(op.body_copy(), qs=qs) * op.repeat_count
            elif isinstance(op, stim.CircuitInstruction):
                if op.name == "TICK":
                    flush()
                    result.append_operation("TICK", [])
                    continue

                if op.name in self.noisy_gates:
                    p = self.noisy_gates[op.name]
                elif self.any_clifford_1 is not None and op.name in ANY_CLIFFORD_1_OPS:
                    p = self.any_clifford_1
                elif self.any_clifford_2 is not None and op.name in ANY_CLIFFORD_2_OPS:
                    p = self.any_clifford_2
                elif op.name in ANNOTATION_OPS:
                    p = 0
                else:
                    raise NotImplementedError(repr(op))
                
                if op.name in SWAP_OPS:
                    self.update_swaps(op)

                pre, mid, post = self.noisy_op(op, p, ancilla)
                current_moment_pre += pre
                current_moment_mid += mid
                current_moment_post += post

                touched_qubits = {
                    t.value for t in op.targets_copy()
                    if t.is_x_target or t.is_y_target or t.is_z_target or t.is_qubit_target
                }
                if op.name in MEASURE_OPS or op.name in RESET_OPS:
                    measured_or_reset_qubits |= touched_qubits
                used_qubits |= touched_qubits
            else:
                raise NotImplementedError(repr(op))
        flush()
        return result

if __name__ == "__main__":
    circuit = stim.Circuit("""
    H 0
    CX 0 1
    M 0 1
    """)
    noisy_gates = {
        "CX": apollo_err_prob["P_CZ"],
        "H": apollo_err_prob["P_SQ"],
        "M": apollo_err_prob["P_READOUT"],
    }
    noisy_gates_connection = {
        "CX": apollo_err_prob["P_CZ"] + 0.3,
    }
    noise = ApolloNoise(0, 0, noisy_gates, noisy_gates_connection)
    noisy = noise.noisy_circuit(circuit)
    print(noisy)

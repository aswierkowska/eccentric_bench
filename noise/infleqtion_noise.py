from typing import Dict, Tuple, Optional, Set
import stim
import math
from .noise import *
from backends import QubitTracking

na_err_prob = {
    "P_CZ": 0.0065, # Derived from CZ fidelity of 99.35(4)% 
    "P_CZ_CROSSTALK": 0.0001, # Based on crosstalk estimation smaller than 10^-4 
    "P_CZ_LEAKAGE": 0.0010, # Model-based estimated leakage probability of 0.10% per gate 
    "P_IDLE": None, # Not explicitly quantified for idle errors
    "P_READOUT": 0.004, # Derived from bright-dark discrimination fidelity of 99.6(2)% 
    "P_RESET": None, # Specific numerical preparation error rate not quantified
    "P_SQ": 0.00098, # Derived from local RZ gate fidelity of 99.902(8)%
    "P_LEAKAGE": 0.009, # State-averaged atom loss probability of 0.9(3)% during NDSSR 
    "P_SHUTTLING_SWAP": 0.0
}

na_gate_times = {
    # TODO: Remove CX
    "CX": 4.16e-7,
    "CZ": 4.16e-7,
    "R": 4.1e-6,
    "H": 2.5e-7,
    "M": 6.0e-3,
}


class InfleqtionNoise(NoiseModel):
    def __init__(
        self,
        idle: float,
        measure_reset_idle: float,
        noisy_gates: Dict[str, float],
        qt: QubitTracking,
        any_clifford_1: Optional[float] = None,
        any_clifford_2: Optional[float] = None,
        use_correlated_parity_measurement_errors: bool = False
    ):
        self.idle = idle
        self.measure_reset_idle = measure_reset_idle
        self.noisy_gates = noisy_gates
        self.qt = qt
        self.any_clifford_1 = any_clifford_1
        self.any_clifford_2 = any_clifford_2
        self.use_correlated_parity_measurement_errors = use_correlated_parity_measurement_errors

    def __repr__(self):
        return f"InfleqtionNoise(idle={self.idle}, measure_reset_idle={self.measure_reset_idle}, gates={list(self.noisy_gates.keys())})"

    @staticmethod
    def get_noise(qt: QubitTracking) -> 'InfleqtionNoise':
        return InfleqtionNoise(
            idle=na_err_prob["P_IDLE"],
            measure_reset_idle=na_err_prob["P_RESET"],
            noisy_gates={
                "CX": na_err_prob["P_CZ"],
                "CZ": na_err_prob["P_CZ"],
                "CZ_CROSSTALK": na_err_prob["P_CZ_CROSSTALK"],
                "CZ_LEAKAGE": na_err_prob["P_CZ_LEAKAGE"],
                "R": na_err_prob["P_RESET"],
                "H": na_err_prob["P_SQ"],
                "M": na_err_prob["P_READOUT"],
                "MPP": na_err_prob["P_READOUT"],
                "SWAP": na_err_prob["P_CZ"],
                "SHUTTLING_SWAP": na_err_prob["P_SHUTTLING_SWAP"],
            },
            qt=qt,
            use_correlated_parity_measurement_errors=True
        )

    def update_swaps(self, op: stim.CircuitInstruction):
        targets = op.targets_copy()
        for i in range(0, len(targets), 2):
            self.qt.swap_qubits(targets[i].qubit_value, targets[i+1].qubit_value)

    def get_qubit_err_prob(self, qubit: int, gate_duration_ns: float = 50) -> float:
        props = self.backend.qubit_properties(qubit)
        t1 = props.t1
        t2 = props.t2
        t = gate_duration_ns
        p_relax = 1 - math.exp(-t / t1)
        p_dephase = 1 - math.exp(-t / t2)
        return (p_relax + p_dephase - p_relax * p_dephase)

    def noisy_op(self, op: stim.CircuitInstruction, base_p: float, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre, mid, post = stim.Circuit(), stim.Circuit(), stim.Circuit()
        targets, args = op.targets_copy(), op.gate_args_copy()

        if op.name in ANY_CLIFFORD_1_OPS:
            for t in targets:
                q = t.value
                gate_time = na_gate_times.get(op.name, 50)
                p = self.get_qubit_err_prob(q, gate_time)
                combined_p = 1 - (1 - base_p) * (1 - p)
                if combined_p > 0:
                    post.append_operation("DEPOLARIZE1", [q], combined_p)
                mid.append_operation(op.name, [t], args)

        elif op.name in ANY_CLIFFORD_2_OPS or op.name in SWAP_OPS:
            for i in range(0, len(targets), 2):
                q1, q2 = targets[i].value, targets[i+1].value
                gate_time = na_gate_times.get(op.name, 50)
                p1 = self.get_qubit_err_prob(q1, gate_time)
                p2 = self.get_qubit_err_prob(q2, gate_time)
                cp1 = 1 - (1 - base_p) * (1 - p1)
                cp2 = 1 - (1 - base_p) * (1 - p2)
                if cp1 > 0: post.append_operation("DEPOLARIZE1", [q1], cp1)
                if cp2 > 0: post.append_operation("DEPOLARIZE1", [q2], cp2)
                mid.append_operation("SWAP" if op.name == "SHUTTLING_SWAP" else op.name, [targets[i], targets[i+1]], args)

        elif op.name in RESET_OPS or op.name in MEASURE_OPS:
            for t in targets:
                q = t.value
                gate_time = na_gate_times.get(op.name, 50)
                p = self.get_qubit_err_prob(q, gate_time)
                combined_p = 1 - (1 - base_p) * (1 - p)
                if combined_p > 0:
                    if op.name in RESET_OPS:
                        post.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", [q], combined_p)
                    if op.name in MEASURE_OPS:
                        pre.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", [q], combined_p)
                mid.append_operation(op.name, [t], args)

        elif op.name == "MPP":
            assert len(targets) % 3 == 0 and all(t.is_combiner for t in targets[1::3]), repr(op)
            for k in range(0, len(targets), 3):
                t1, t2 = targets[k], targets[k+2]
                base_p = self.get_gate_error(stim.CircuitInstruction(op.name, [t1, targets[k+1], t2], args))
                p1 = self.get_qubit_err_prob(t1.value)
                p2 = self.get_qubit_err_prob(t2.value)
                cp = 1 - (1 - base_p) * (1 - (p1 + p2) / 2)
                if self.use_correlated_parity_measurement_errors:
                    mid += parity_measurement_with_correlated_measurement_noise(t1=t1, t2=t2, ancilla=ancilla, mix_probability=cp)
                else:
                    pre.append_operation("DEPOLARIZE2", [t1.value, t2.value], cp)
                mid.append_operation(op.name, [t1, targets[k+1], t2], args)

        else:
            raise NotImplementedError(repr(op))

        return pre, mid, post

    def noisy_circuit(self, circuit: stim.Circuit, *, qs: Optional[Set[int]] = None) -> stim.Circuit:
        result = stim.Circuit()
        ancilla = circuit.num_qubits
        current_moment_pre, current_moment_mid, current_moment_post = stim.Circuit(), stim.Circuit(), stim.Circuit()
        used_qubits, measured_or_reset_qubits = set(), set()
        qs = qs or set(range(circuit.num_qubits))

        def flush():
            if not current_moment_mid:
                return
            idle_qs = sorted(qs - used_qubits)
            if used_qubits and idle_qs and self.idle > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qs, self.idle)
            if measured_or_reset_qubits:
                idle_qs = sorted(qs - measured_or_reset_qubits)
                if idle_qs and self.measure_reset_idle > 0:
                    current_moment_post.append_operation("DEPOLARIZE1", idle_qs, self.measure_reset_idle)
            nonlocal result
            result += current_moment_pre + current_moment_mid + current_moment_post
            current_moment_pre.clear()
            current_moment_mid.clear()
            current_moment_post.clear()
            used_qubits.clear()
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
                elif self.any_clifford_1 and op.name in ANY_CLIFFORD_1_OPS:
                    p = self.any_clifford_1
                elif self.any_clifford_2 and op.name in ANY_CLIFFORD_2_OPS:
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

                touched = {t.value for t in op.targets_copy() if t.is_qubit_target}
                if op.name in MEASURE_OPS or op.name in RESET_OPS:
                    measured_or_reset_qubits |= touched
                used_qubits |= touched
            else:
                raise NotImplementedError(repr(op))
        flush()
        return result


# Quick test
if __name__ == "__main__":
    circuit = stim.Circuit("H 0\nCX 0 1\nM 0 1")
    qt = QubitTracking(num_qubits=2)

    class DummyBackend:
        def qubit_properties(self, qubit):
            return type("QProps", (), {"t1": 1e6, "t2": 1e6})()

    noise = InfleqtionNoise.get_noise(qt, backend=DummyBackend())
    noisy = noise.noisy_circuit(circuit)
    print(noisy)

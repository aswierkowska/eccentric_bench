from typing import Dict, Tuple, Optional, Set
import stim
import random
import math
import numpy as np
from .noise import *
from backends import FakeIBMFlamingo, QubitTracking
from qiskit.providers import QubitProperties

flamingo_err_prob = {
    "P_RESET": 0.0,
    "P_SQ": 0.00025,
    "P_TQ": 0.002,
    "P_MEASUREMENT": 0.01,
    "P_READOUT": 8.057e-3,
    "P_IDLE": 0.0
}

flamingo_gate_times = {
    "SQ": 50 * 1e-9,
    "TQ": 70 * 1e-9,
    "M": 70 * 1e-9,
}

P_CROSSTALK = 0.0

class FlamingoNoise(NoiseModel):
    def __init__(
        self,
        idle: float,
        measure_reset_idle: float,
        noisy_gates: Dict[str, float],
        noisy_gates_connection: Dict[str, float],
        qt: QubitTracking,
        backend: FakeIBMFlamingo,
        any_clifford_1: Optional[float] = None,
        any_clifford_2: Optional[float] = None,
        use_correlated_parity_measurement_errors: bool = False
    ):
        self.idle = idle
        self.measure_reset_idle = measure_reset_idle
        self.noisy_gates = noisy_gates
        self.noisy_gates_connection = noisy_gates_connection
        self.qt = qt
        self.backend = backend
        self.any_clifford_1 = any_clifford_1
        self.any_clifford_2 = any_clifford_2
        self.use_correlated_parity_measurement_errors = use_correlated_parity_measurement_errors
        self.crosstalk_prob = 0.0 # TODO: set value

    @staticmethod
    def get_noise(
        qt: QubitTracking,
        backend: FakeIBMFlamingo
    ) -> 'FlamingoNoise':
        return FlamingoNoise(
            idle=flamingo_err_prob["P_IDLE"],
            measure_reset_idle=flamingo_err_prob["P_MEASUREMENT"],
            noisy_gates={
                "CX": flamingo_err_prob["P_TQ"],
                "CZ": flamingo_err_prob["P_TQ"],
                "SWAP": flamingo_err_prob["P_TQ"],
                "R": flamingo_err_prob["P_SQ"],
                "H": flamingo_err_prob["P_SQ"],
                "M": flamingo_err_prob["P_MEASUREMENT"],
                "MPP": flamingo_err_prob["P_READOUT"],
                "RESET": flamingo_err_prob["P_RESET"]
            },
            noisy_gates_connection={
                "CX": 0.03,
                "CZ": 0.03,
                "SWAP": 0.03,
            },
            qt=qt,
            backend=backend,
            use_correlated_parity_measurement_errors=True
        )
    
    def add_crosstalk_errors_for_moment(self, used_qubits: Set[int], post: stim.Circuit, p: float):
        already_noised = set()
        for q in used_qubits:
            for neighbor in self.qt.get_neighbours(q):
                if neighbor in used_qubits and neighbor not in already_noised:
                    if random.random() < p:
                        noise_op = "X_ERROR" if random.random() < 0.5 else "Z_ERROR"
                        post.append_operation(noise_op, [stim.target_qubit(neighbor)], 1.0)
                        already_noised.add(neighbor)

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
    
    def get_gate_time(self, op: stim.CircuitInstruction) -> float:
        name = op.name           
        if name in flamingo_gate_times:
            return flamingo_gate_times["TQ"]
        raise NotImplementedError(f"Gate time not defined for op: {repr(op)}")

    def get_gate_error(self, op: stim.CircuitInstruction) -> float:
        name = op.name
        if name in self.noisy_gates:
            return self.noisy_gates[name]
        raise NotImplementedError(f"Gate error not defined for op: {repr(op)}")

    def noisy_op(self, op: stim.CircuitInstruction, base_p: float, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()

        if op.name in ANY_CLIFFORD_1_OPS:
            for t in targets:
                q = t.value
                base_p = self.get_gate_error(stim.CircuitInstruction(op.name, [t], args))
                qubit_p = self.get_qubit_err_prob(q, flamingo_gate_times["SQ"])
                combined_p = 1 - (1 - base_p) * (1 - qubit_p)
                if combined_p > 0:
                    post.append_operation("DEPOLARIZE1", [q], combined_p)
                mid.append_operation(op.name, [t], args)

        elif op.name in ANY_CLIFFORD_2_OPS or op.name in SWAP_OPS:
            for i in range(0, len(targets), 2):
                q1 = targets[i].value
                q2 = targets[i+1].value
                base_p = self.get_gate_error(stim.CircuitInstruction(op.name, [targets[i], targets[i+1]], args))
                p1 = self.get_qubit_err_prob(q1, self.get_gate_time(op))
                p2 = self.get_qubit_err_prob(q2, self.get_gate_time(op))
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
                base_p = self.get_gate_error(stim.CircuitInstruction(op.name, [t], args))
                qubit_p = self.get_qubit_err_prob(q, flamingo_gate_times["M"])
                combined_p = 1 - (1 - base_p) * (1 - qubit_p)
                if combined_p > 0:
                    if op.name in RESET_OPS:
                        post.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", [q], combined_p)
                    if op.name in MEASURE_OPS:
                        pre.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", [q], combined_p)
                mid.append_operation(op.name, [t], args)

        elif op.name == "MPP":
            # TODO: CLEAN AND TEST
            assert len(targets) % 3 == 0 and all(t.is_combiner for t in targets[1::3]), repr(op)
            assert args == [] or args == [0]
            for k in range(0, len(targets), 3):
                t1 = targets[k]
                t2 = targets[k + 2]
                mini_op = stim.CircuitInstruction(op.name, [t1, targets[k+1], t2], args)
                base_p = self.get_gate_error(mini_op)
                p1 = self.get_qubit_err_prob(t1.value)
                p2 = self.get_qubit_err_prob(t2.value)
                avg_qubit_p = (p1 + p2) / 2
                combined_p = 1 - (1 - base_p) * (1 - avg_qubit_p)
                if self.use_correlated_parity_measurement_errors:
                    mid += parity_measurement_with_correlated_measurement_noise(
                        t1=t1,
                        t2=t2,
                        ancilla=ancilla,
                        mix_probability=combined_p
                    )
                else:
                    pre.append_operation("DEPOLARIZE2", [t1.value, t2.value], combined_p)
                mid.append_operation(op.name, [t1, targets[k+1], t2], args)
            return pre, mid, post

        else:
            raise NotImplementedError(repr(op))

        return pre, mid, post


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

            if self.crosstalk_prob > 0:
                self.add_crosstalk_errors_for_moment(used_qubits, current_moment_post, self.crosstalk_prob)

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

                p = None

                if op.name in self.noisy_gates:
                    p = self.noisy_gates[op.name]
                elif self.any_clifford_1 is not None and op.name in ANY_CLIFFORD_1_OPS:
                    p = self.any_clifford_1
                elif self.any_clifford_2 is not None and (op.name in ANY_CLIFFORD_2_OPS or op.name in SWAP_OPS):
                    p = self.any_clifford_2
                elif op.name in ANNOTATION_OPS:
                    flush()
                    result.append_operation(op.name, op.targets_copy(), op.gate_args_copy())
                    continue
                if p == None:
                    raise NotImplementedError(repr(op))

                pre, mid, post = self.noisy_op(op, p, ancilla)

                if op.name in SWAP_OPS:
                    self.update_swaps(op)

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

import stim
from typing import Dict, Optional, Set, Tuple
from .noise import *

# From https://arxiv.org/pdf/2408.13687
willow_err_prob = {
    "P_CZ": 2.8e-3,
    "P_CZ_CROSSTALK": 5.5e-4,
    "P_CZ_LEAKAGE": 2.0e-4,
    "P_IDLE": 0.9e-2,
    "P_READOUT": 0.8e-2,
    "P_RESET": 1.5e-3,
    "P_SQ": 6.2e-4,
    "P_LEAKAGE": 2.5e-4
}

"""
# From https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf
willow_err_prob = {
    "P_CZ": 0.33e-2, # Two-qubit gate error (mean, simultaneous) for CZ [1]
    "P_CZ_CROSSTALK": None,
    "P_CZ_LEAKAGE": None,
    "P_IDLE": None,
    "P_READOUT": 0.77e-2,
    "P_RESET": None,
    "P_SQ": 0.035e-2, # Single-qubit gate error (mean, simultaneous) [1]
    "P_LEAKAGE": None
}
"""


class WillowNoise(NoiseModel):
    def __init__(self, qt):
        super().__init__(
            idle=willow_err_prob["P_IDLE"],
            measure_reset_idle=willow_err_prob["P_RESET"],
            noisy_gates={
                # TODO There should not be CX
                "CX": willow_err_prob["P_CZ"],
                "CZ": willow_err_prob["P_CZ"],
                "CZ_CROSSTALK": willow_err_prob["P_CZ_CROSSTALK"],
                "CZ_LEAKAGE": willow_err_prob["P_CZ_LEAKAGE"],
                "R": willow_err_prob["P_RESET"],
                # TODO: Should not be H
                "H": willow_err_prob["P_SQ"],
                "M": willow_err_prob["P_READOUT"],
                "MPP": willow_err_prob["P_READOUT"],
            },
            use_correlated_parity_measurement_errors=True
        )
        self.qt = qt

    @staticmethod
    def get_noise(qt) -> 'WillowNoise':
        return WillowNoise(
            qt=qt,
            idle=willow_err_prob["P_IDLE"],
            measure_reset_idle=willow_err_prob["P_RESET"],
            noisy_gates={
                "CX": willow_err_prob["P_CZ"],
                "CZ": willow_err_prob["P_CZ"],
                "CZ_CROSSTALK": willow_err_prob["P_CZ_CROSSTALK"],
                "CZ_LEAKAGE": willow_err_prob["P_CZ_LEAKAGE"],
                "R": willow_err_prob["P_RESET"],
                "H": willow_err_prob["P_SQ"],
                "M": willow_err_prob["P_READOUT"],
                "MPP": willow_err_prob["P_READOUT"],
            },
            noisy_gates_connection={
                "CX": willow_err_prob["P_CZ"] + 0.3,
            },
            use_correlated_parity_measurement_errors=True
        )

    def update_swaps(self, op: stim.CircuitInstruction):
        targets = op.targets_copy()
        for i in range(0, len(targets), 2):
            q1 = targets[i].qubit_value
            q2 = targets[i+1].qubit_value
            self.qt.swap_qubits(q1, q2)

    def noisy_op(self, op: stim.CircuitInstruction, p: float, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if p > 0:
            if op.name in ANY_CLIFFORD_1_OPS:
                post.append_operation("DEPOLARIZE1", targets, p)
            elif op.name in ANY_CLIFFORD_2_OPS:
                post.append_operation("DEPOLARIZE2", targets, p)
            elif op.name in RESET_OPS or op.name in MEASURE_OPS:
                if op.name in RESET_OPS:
                    post.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, p)
                if op.name in MEASURE_OPS:
                    pre.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, p)
            elif op.name == "MPP":
                assert len(targets) % 3 == 0 and all(t.is_combiner for t in targets[1::3]), repr(op)
                assert args == [] or args == [0]

                if self.use_correlated_parity_measurement_errors:
                    for k in range(0, len(targets), 3):
                        mid += parity_measurement_with_correlated_measurement_noise(
                            t1=targets[k],
                            t2=targets[k + 2],
                            ancilla=ancilla,
                            mix_probability=p
                        )
                    return pre, mid, post
                else:
                    pre.append_operation("DEPOLARIZE2", [t.value for t in targets if not t.is_combiner], p)
            else:
                raise NotImplementedError(repr(op))
        mid.append_operation(op.name, targets, args)
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
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
    
    def get_gate_time(self, op: stim.CircuitInstruction) -> float:
        name = op.name
        if name in self.noisy_gates_connection and len(op.target_groups()[0]) >= 2:
            logical_q1 = op.target_groups()[0][0].qubit_value
            logical_q2 = op.target_groups()[0][1].qubit_value
            phy_q1 = self.qt.get_layout_postion(logical_q1)
            phy_q2 = self.qt.get_layout_postion(logical_q2)

            if (phy_q1, phy_q2) in self.backend.get_remote_gates or (phy_q2, phy_q1) in self.backend.get_remote_gates:
                duration = 300
                dt = 2.2222222222222221e-10 * 1e9
                rounded_duration = round((duration * 1e-9) / dt) * dt
                return rounded_duration
    
            
        if name in flamingo_gate_times:
            return flamingo_gate_times["TQ"]
        raise NotImplementedError(f"Gate time not defined for op: {repr(op)}")

    def get_gate_error(self, op: stim.CircuitInstruction) -> float:
        # TODO: follow this https://github.com/manosgior/HybridDQC/blob/main/backends/backend.py#L159
        name = op.name
        if name in self.noisy_gates_connection and len(op.target_groups()[0]) >= 2:
            logical_q1 = op.target_groups()[0][0].qubit_value
            logical_q2 = op.target_groups()[0][1].qubit_value
            phy_q1 = self.qt.get_layout_postion(logical_q1)
            phy_q2 = self.qt.get_layout_postion(logical_q2)

            if (phy_q1, phy_q2) in self.backend.get_remote_gates or (phy_q2, phy_q1) in self.backend.get_remote_gates:
                return self.noisy_gates_connection[name]
            
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
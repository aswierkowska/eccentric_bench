from typing import Tuple
import stim
from .noise import *
from backends import QubitTracking

apollo_err_prob = {
    "P_SQ": 0.000002,
    "P_TQ": 0.0001,
    "P_MEASUREMENT": 0.0001,
    "P_SHUTTLING_SWAP": 0,
    "P_SWAP": 0,
    "P_IDLE": 0,
    "P_RESET": 0,
    "P_READOUT": 0,
}

P_MEMORY_ERROR = 0.00223

# TODO: add time -- t1 few minutes
apollo_gate_times = {
    "RESET": 0.0,
    "SQ": 0,
    "TQ": 0,
    "M": 0,
    "SHUTTLING_SWAP": 0,
    "SWAP": 0,
}

P_CROSSTALK = 0.66e-6

class ApolloNoise(NoiseModel):
    @staticmethod
    def get_noise(qt: QubitTracking) -> 'NoiseModel':
        return NoiseModel(
            idle=apollo_err_prob["P_IDLE"],
            measure_reset_idle=apollo_err_prob["P_RESET"],
            crosstalk_gates={
                "R": 0.04567e-4,
                "SQ": 0.066e-5,
                "TQ": 0.066e-5,
                "M": 0.03867e-4,
            },
            noisy_gates={
                "CX": apollo_err_prob["P_TQ"],
                "CZ": apollo_err_prob["P_TQ"],
                "SWAP": apollo_err_prob["P_TQ"],
                "R": apollo_err_prob["P_RESET"],
                "H": apollo_err_prob["P_SQ"],
                "M": apollo_err_prob["P_MEASUREMENT"],
                "MPP": apollo_err_prob["P_READOUT"],
                "SHUTTLING_SWAP": apollo_err_prob["P_SHUTTLING_SWAP"],
            },
            qt=qt,
            use_correlated_parity_measurement_errors=True
        )

    def noisy_op(self, op: stim.CircuitInstruction, base_p: float, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        # TODO now annotations can be passed with p = 0
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()

        if op.name in ANY_CLIFFORD_1_OPS:
            for t in targets:
                q = t.value
                if base_p > 0:
                    post.append_operation("DEPOLARIZE1", [q], base_p)
                mid.append_operation(op.name, [t], args)

        elif op.name in ANY_CLIFFORD_2_OPS or op.name in SWAP_OPS:
            for i in range(0, len(targets), 2):
                q1 = targets[i].value
                q2 = targets[i+1].value
                if base_p > 0:
                    post.append_operation("DEPOLARIZE2", [q1, q2], base_p)
                if op.name == "SHUTTLING_SWAP":
                    post.append_operation("DEPOLARIZE2", [q1, q2], P_MEMORY_ERROR)
                    mid.append_operation("SWAP", [targets[i], targets[i+1]], args)
                else:
                    mid.append_operation(op.name, [targets[i], targets[i+1]], args)

        elif op.name in RESET_OPS or op.name in MEASURE_OPS:
            for t in targets:
                q = t.value
                qubit_p = self.get_qubit_err_prob(q, apollo_gate_times[op.name])
                combined_p = 1 - (1 - base_p) * (1 - qubit_p)
                if base_p > 0:
                    if op.name in RESET_OPS:
                        self.add_crosstalk_error(op, post, 0.04567e-4)
                        post.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", [q], combined_p)
                    if op.name in MEASURE_OPS:
                        self.add_crosstalk_error(op, post, 0.03867e-4)
                        pre.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", [q], combined_p)
                mid.append_operation(op.name, [t], args)

        return pre, mid, post
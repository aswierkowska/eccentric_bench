from .noise import *
from backends import QubitTracking

# TODO: Should the crosstalk be more granular?
# crosstalk_gates={
#    "R": 0.04567e-4,
#    "SQ": 0.066e-5,
#    "TQ": 0.066e-5,
#    "M": 0.03867e-4,
#}

class ApolloNoise(NoiseModel):

    @staticmethod
    def get_noise(qt: QubitTracking) -> 'NoiseModel':
        # TODO: get more detailed values from https://github.com/CQCL/quantinuum-hardware-specifications/blob/main/qtm_spec/combined_analysis.py or here: https://arxiv.org/pdf/2406.02501
        # Values taken from Quantinuum H2 and rescaled according to their roadmap
        return NoiseModel(
            sq=0.000002,
            tq=0.0001,
            idle=0.00223,
            crosstalk=0.66e-6,
            measure=0.0001,
            qt=qt,
            # backend seems unnecessary as t1 is a few minutes
        )
    """
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
    """
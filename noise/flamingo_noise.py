from .noise import *
from backends import FakeIBMFlamingo, QubitTracking


class FlamingoNoise(NoiseModel):

    @staticmethod
    def get_noise(
        qt: QubitTracking,
        backend: FakeIBMFlamingo
    ) -> 'NoiseModel':
        return NoiseModel(
            sq=0.00025,
            tq=0.002,
            measure=0.01,
            remote=0.03,
            gate_times={
                # TODO: what is reset gate time?
                "SQ": 50 * 1e-9,
                "TQ": 70 * 1e-9,
                "M": 70 * 1e-9,
                "REMOTE": round((300 * 1e-9) / (2.2222222222222221e-10 * 1e9)) * (2.2222222222222221e-10 * 1e9),
            },
            qt=qt,
            backend=backend
        )

    """
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
    """
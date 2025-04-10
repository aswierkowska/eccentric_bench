import dataclasses
import stim
from noise import *
from backends import get_layout_postion, FakeIBMFlamingo

# From https://arxiv.org/pdf/2408.13687
flamingo_err_prob = {
    "P_CZ": 0,
    "P_CZ_CROSSTALK": 0,
    "P_CZ_LEAKAGE": 0,
    "P_IDLE": 0,
    "P_READOUT": 0,
    "P_RESET": 0,
    "P_SQ": 0,
    "P_LEAKAGE": 0
}

@dataclasses.dataclass(frozen=True)
class FlamingoNoise(NoiseModel):
    noisy_gates_connection: Dict[str, float]
    backend: FakeIBMFlamingo

    @staticmethod
    def get_noise(
        noisy_gates: Dict[str, float], 
        noisy_gates_connection: Dict[str, float],
        backend: FakeIBMFlamingo
    ) -> 'FlamingoNoise':
        # Default to the predefined probabilities if no specific gates or connection is provided
        if noisy_gates is None:
            noisy_gates = {
                "CX": flamingo_err_prob["P_CZ"],
                "H": flamingo_err_prob["P_SQ"],
                "M": flamingo_err_prob["P_READOUT"],
            }

        if noisy_gates_connection is None:
            noisy_gates_connection = {
                "CX": flamingo_err_prob["P_CZ"] + 0.3,
            }

        # Create a FlamingoNoise instance with the provided or default noisy gates and backend
        return FlamingoNoise(
           idle=flamingo_err_prob["P_IDLE"],
            measure_reset_idle=flamingo_err_prob["P_RESET"],
            noisy_gates={
                # TODO: should not be CX
                "CX": flamingo_err_prob["P_CZ"],
                "CZ": flamingo_err_prob["P_CZ"],
                "CZ_CROSSTALK": flamingo_err_prob["P_CZ_CROSSTALK"],
                "CZ_LEAKAGE": flamingo_err_prob["P_CZ_LEAKAGE"],
                "R": flamingo_err_prob["P_RESET"],
                # TODO: should not be H
                "H": flamingo_err_prob["P_SQ"],
                "M": flamingo_err_prob["P_READOUT"],
                "MPP": flamingo_err_prob["P_READOUT"],
            },
            noisy_gates_connection = {
                "CX": flamingo_err_prob["P_CZ"] + 0.3,
            },
            backend = backend,
            use_correlated_parity_measurement_errors=True
        )




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
                            mix_probability=p)
                    return pre, mid, post

                else:
                    pre.append_operation("DEPOLARIZE2", [t.value for t in targets if not t.is_combiner], p)
                    args = [p]

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

            # Apply idle depolarization rules.
            idle_qubits = sorted(qs - used_qubits)
            if used_qubits and idle_qubits and self.idle > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.idle)
            idle_qubits = sorted(qs - measured_or_reset_qubits)
            if measured_or_reset_qubits and idle_qubits and self.measure_reset_idle > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.measure_reset_idle)

            # Move current noisy moment into result.
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

                if len(op.target_groups()[0]) > 1:
                    logical_q1 = op.target_groups()[0][0].qubit_value
                    logical_q2 = op.target_groups()[0][1].qubit_value
                    phy_q1 = get_layout_postion(logical_q1)
                    phy_q2 = get_layout_postion(logical_q2)
                    if (phy_q1, phy_q2) in backend.get_remote_gates():
                        p = self.noisy_gates_conn[op.name]

                if p is None and op.name in self.noisy_gates:
                    p = self.noisy_gates[op.name]
                elif self.any_clifford_1 is not None and op.name in ANY_CLIFFORD_1_OPS:
                    p = self.any_clifford_1
                elif self.any_clifford_2 is not None and op.name in ANY_CLIFFORD_2_OPS:
                    p = self.any_clifford_2
                elif op.name in ANNOTATION_OPS:
                    p = 0
                else:
                    raise NotImplementedError(repr(op))
                pre, mid, post = self.noisy_op(op, p, ancilla)
                current_moment_pre += pre
                current_moment_mid += mid
                current_moment_post += post

                # Ensure the circuit is not touching qubits multiple times per tick.
                touched_qubits = {
                    t.value
                    for t in op.targets_copy()
                    if t.is_x_target or t.is_y_target or t.is_z_target or t.is_qubit_target
                }
                if op.name in ANNOTATION_OPS:
                    touched_qubits.clear()
                # Hack: turn off this assertion off for now since correlated errors are built into circuit.
                #assert touched_qubits.isdisjoint(used_qubits), repr(current_moment_pre + current_moment_mid + current_moment_post)
                used_qubits |= touched_qubits
                if op.name in MEASURE_OPS or op.name in RESET_OPS:
                    measured_or_reset_qubits |= touched_qubits
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
    noisy_gates={
                "CX": flamingo_err_prob["P_CZ"],
                "H": flamingo_err_prob["P_SQ"],
                "M": flamingo_err_prob["P_READOUT"],
    }
    noisy_gates_connection={
        "CX": flamingo_err_prob["P_CZ"] + 0.3,
    }
    noise = FlamingoNoise(0, 0, noisy_gates, noisy_gates_connection)
    noise.noisy_circuit(circuit)
    #plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="flamingo.png")
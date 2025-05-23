from qiskit.providers import BackendV2
from backends import QubitTracking

import random
import math

#################################################################################################################
# Adapted from: https://github.com/Strilanc/honeycomb_threshold/blob/main/src/noise.py
#################################################################################################################
from typing import Optional, Dict, Set, Tuple

import stim

SQ_OPS = {"C_XYZ", "C_ZYX", "H", "H_YZ", "I", "X"}
TQ_OPS = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
RESET_OPS = {"R", "RX", "RY"}
MEASURE_OPS = {"M", "MX", "MY"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}
SWAP_OPS = {"SWAP"}

class NoiseModel:
    def __init__(
        self,
        sq: Optional[float] = 0,
        tq: Optional[float] = 0,
        idle: Optional[float] = 0,
        crosstalk: Optional[float] = 0,
        leakage: Optional[float] = 0,
        reset: Optional[float] = 0,
        measure: Optional[float] = 0,
        shuttle: Optional[float] = 0,
        remote: Optional[float] = 0,
        noisy_gates: Dict[str, float] = None,
        gate_times: Optional[Dict[str, float]] = None,
        qt: Optional[QubitTracking] = None,
        backend: Optional[BackendV2] = None,
        use_correlated_parity_measurement_errors: bool = False
    ):
        self.sq = sq
        self.tq = tq
        self.idle = idle
        self.crosstalk = crosstalk
        self.leakage = leakage
        self.reset = reset
        self.measure = measure
        self.shuttle = shuttle
        self.remote = remote
        self.noisy_gates = noisy_gates
        self.gate_times = gate_times
        self.qt = qt
        self.backend = backend
        self.use_correlated_parity_measurement_errors = use_correlated_parity_measurement_errors

    @staticmethod
    def SD6(p: float) -> 'NoiseModel':
        return NoiseModel(
            sq=p,
            idle=p,
            measure=0,
            reset=0,
            noisy_gates={
                "CX": p,
                "R": p,
                "M": p,
            },
        )

    @staticmethod
    def PC3(p: float) -> 'NoiseModel':
        return NoiseModel(
            sq=p,
            tq=p,
            idle=p,
            measure=0,
            reset=0,
            noisy_gates={
                "R": p,
                "M": p,
            },
        )

    @staticmethod
    def EM3_v1(p: float) -> 'NoiseModel':
        """EM3 but with measurement flip errors independent of measurement target depolarization error."""
        return NoiseModel(
            idle=p,
            measure=0,
            reset=0,
            sq=p,
            noisy_gates={
                "R": p,
                "M": p,
                "MPP": p,
            },
        )

    @staticmethod
    def EM3_v2(p: float) -> 'NoiseModel':
        """EM3 with measurement flip errors correlated with measurement target depolarization error."""
        return NoiseModel(
            sq=0,
            tq=0,
            idle=p,
            measure=0,
            reset=0,
            use_correlated_parity_measurement_errors=True,
            noisy_gates={
                "R": p/2,
                "M": p/2,
                "MPP": p,
            },
        )

    @staticmethod
    def SI1000(p: float) -> 'NoiseModel':
        """Inspired by superconducting device."""
        return NoiseModel(
            sq=p / 10,
            idle=p / 10,
            measure=2 * p,
            reset=2 * p,
            noisy_gates={
                "CZ": p,
                "R": 2 * p,
                "M": 5 * p,
            },
        )
    
    def update_swaps(self, op: stim.CircuitInstruction):
        targets = op.targets_copy()
        for i in range(0, len(targets), 2):
            q1 = targets[i].qubit_value
            q2 = targets[i+1].qubit_value
            self.qt.swap_qubits(q1, q2)

    # TODO: overlaps with idle errors?
    def get_qubit_err_prob(self, qubit: int, gate_duration_ns: float) -> float:
        qubit_properties = self.backend.qubit_properties(qubit)
        t1 = qubit_properties.t1
        t2 = qubit_properties.t2
        t = gate_duration_ns
        p_relax = 1 - math.exp(-t / t1)
        p_dephase = 1 - math.exp(-t / t2)
        return (p_relax + p_dephase - p_relax * p_dephase)


    def add_crosstalk_errors_for_moment(self, touched_qubits: set[int], post: stim.Circuit, gate: str):
        p = self.crosstalk_gates.get(gate, 0)
        if gate in SQ_OPS:
            p = self.crosstalk_gates.get("SQ", p)
        elif gate in TQ_OPS:
            p = self.crosstalk_gates.get("TQ", p)

        already_noised = set()
        for q in touched_qubits:
            for neighbor in self.qt.get_neighbours(q):
                if neighbor in touched_qubits and (neighbor, q) not in already_noised and (q, neighbor) not in already_noised:
                    if random.random() < p:
                        noise_op = "X_ERROR" if random.random() < 0.5 else "Z_ERROR"
                        post.append_operation(noise_op, [stim.target_qubit(neighbor)], 1.0)
                        already_noised.add((q, neighbor))
                        already_noised.add((neighbor, q))
    
    def get_gate_time(self, op: stim.CircuitInstruction) -> float:
        name = op.name
        if name in self.noisy_gates_connection and len(op.target_groups()[0]) >= 2:
            logical_q1 = op.target_groups()[0][0].qubit_value
            logical_q2 = op.target_groups()[0][1].qubit_value
            phy_q1 = self.qt.get_layout_postion(logical_q1)
            phy_q2 = self.qt.get_layout_postion(logical_q2)

            if (phy_q1, phy_q2) in self.backend.get_remote_gates or (phy_q2, phy_q1) in self.backend.get_remote_gates:
                if name == "SWAP" and (self.backend.name == "FakeQuantinuumApollo" or self.backend.name == "FakeInfleqtion"):
                    return self.gate_times["REMOTE"] * self.qt.get_euclidian_distance(phy_q1, phy_q2)
                else:
                    return self.gate_times["REMOTE"]
            
        if name in self.gate_times:
            return self.gate_times["TQ"]
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

    # TODO: do it IDLE then IDLE if not function to get t1/t2 from backend
    def noisy_op(self, op: stim.CircuitInstruction, p: float, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if p > 0:
            if op.name in SQ_OPS:
                post.append_operation("DEPOLARIZE1", targets, p)
            # TODO: modified by adding SWAP_OPS
            elif op.name in TQ_OPS or op.name in SWAP_OPS:
                for i in range(0, len(targets), 2):
                    pair = [targets[i], targets[i+1]]
                    post.append_operation("DEPOLARIZE2", pair, p)
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
        measured_qubits: Set[int] = set()
        reset_qubits: Set[int] = set()
        if qs is None:
            qs = set(range(circuit.num_qubits))

        def flush():
            nonlocal result
            if not current_moment_mid:
                return

            # TODO: this kinda covers what qt is for
            # Apply idle depolarization rules.
            idle_qubits = sorted(qs - used_qubits)
            if used_qubits and idle_qubits and self.idle > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.idle)
            idle_qubits = sorted(qs - measured_qubits)
            if measured_qubits and idle_qubits and self.measure > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.measure)
            idle_qubits = sorted(qs - reset_qubits)
            if reset_qubits and idle_qubits and self.reset > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.reset)

            # Move current noisy moment into result.
            result += current_moment_pre
            result += current_moment_mid
            result += current_moment_post
            used_qubits.clear()
            current_moment_pre.clear()
            current_moment_mid.clear()
            current_moment_post.clear()
            measured_qubits.clear()
            reset_qubits.clear()

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
                elif self.sq is not None and op.name in SQ_OPS:
                    p = self.sq
                elif self.tq is not None and (op.name in TQ_OPS or op.name in SWAP_OPS):
                    p = self.tq
                elif op.name in ANNOTATION_OPS:
                    p = 0
                    # TODO: Modified not to enter noisy_op in case it's just annotation
                    # flush()
                    # result.append_operation(op.name, op.targets_copy(), op.gate_args_copy())
                    # continue
                else:
                    raise NotImplementedError(repr(op))
                
                if op.name in SWAP_OPS:
                    self.update_swaps(op)

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
                
                # Modification
                self.add_crosstalk_errors_for_moment(touched_qubits, post, op.name)
                
                used_qubits |= touched_qubits
                if op.name in MEASURE_OPS:
                    measured_qubits |= touched_qubits
                if op.name in RESET_OPS:
                    reset_qubits |= touched_qubits
            else:
                raise NotImplementedError(repr(op))
        flush()

        return result


def mix_probability_to_independent_component_probability(mix_probability: float, n: float) -> float:
    """Converts the probability of applying a full mixing channel to independent component probabilities.

    If each component is applied independently with the returned component probability, the overall effect
    is identical to, with probability `mix_probability`, uniformly picking one of the components to apply.

    Not that, unlike in other places in the code, the all-identity case is one of the components that can
    be picked when applying the error case.
    """
    return 0.5 - 0.5 * (1 - mix_probability) ** (1 / 2 ** (n - 1))


def parity_measurement_with_correlated_measurement_noise(
        *,
        t1: stim.GateTarget,
        t2: stim.GateTarget,
        ancilla: int,
        mix_probability: float) -> stim.Circuit:
    """Performs a noisy parity measurement.

    With probability mix_probability, applies a random element from

        {I1,X1,Y1,Z1}*{I2,X2,Y2,Z2}*{no flip, flip}

    Note that, unlike in other places in the code, the all-identity term is one of the possible
    samples when the error occurs.
    """

    ind_p = mix_probability_to_independent_component_probability(mix_probability, 5)

    # Generate all possible combinations of (non-identity) channels.  Assumes triple of targets
    # with last element corresponding to measure qubit.
    circuit = stim.Circuit()
    circuit.append_operation('R', [ancilla])
    if t1.is_x_target:
        circuit.append_operation('XCX', [t1.value, ancilla])
    if t1.is_y_target:
        circuit.append_operation('YCX', [t1.value, ancilla])
    if t1.is_z_target:
        circuit.append_operation('ZCX', [t1.value, ancilla])
    if t2.is_x_target:
        circuit.append_operation('XCX', [t2.value, ancilla])
    if t2.is_y_target:
        circuit.append_operation('YCX', [t2.value, ancilla])
    if t2.is_z_target:
        circuit.append_operation('ZCX', [t2.value, ancilla])

    first_targets = ["I", stim.target_x(t1.value), stim.target_y(t1.value), stim.target_z(t1.value)]
    second_targets = ["I", stim.target_x(t2.value), stim.target_y(t2.value), stim.target_z(t2.value)]
    measure_targets = ["I", stim.target_x(ancilla)]

    errors = []
    for first_target in first_targets:
        for second_target in second_targets:
            for measure_target in measure_targets:
                error = []
                if first_target != "I":
                    error.append(first_target)
                if second_target != "I":
                    error.append(second_target)
                if measure_target != "I":
                    error.append(measure_target)

                if len(error) > 0:
                    errors.append(error)

    for error in errors:
        circuit.append_operation("CORRELATED_ERROR", error, ind_p)

    circuit.append_operation('M', [ancilla])

    return circuit
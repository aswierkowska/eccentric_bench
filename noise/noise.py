from qiskit.providers import BackendV2
from backends import QubitTracking

import random
import math

#################################################################################################################
# Adapted from: https://github.com/Strilanc/honeycomb_threshold/blob/main/src/noise.py
#################################################################################################################
from typing import Optional, Dict, Set, Tuple

import stim

ANY_CLIFFORD_1_OPS = {"C_XYZ", "C_ZYX", "H", "H_YZ", "I", "X"}
ANY_CLIFFORD_2_OPS = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
RESET_OPS = {"R", "RX", "RY"}
MEASURE_OPS = {"M", "MX", "MY"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}
SWAP_OPS = {"SWAP", "SHUTTLING_SWAP"} # TODO: is shuttling_swap still needed?

#@dataclasses.dataclass(frozen=True)
class NoiseModel:
    idle: float #idle should be either float or like a Dict to map qubits to their respective idle noise
    measure_reset_idle: float
    crosstalk_gates: Dict[str, float]
    noisy_gates: Dict[str, float]
    any_clifford_1: Optional[float] = None
    any_clifford_2: Optional[float] = None
    use_correlated_parity_measurement_errors: bool = False

    def __init__(
        self,
        idle: float,
        measure_reset_idle: float,
        crosstalk_gates: Dict[str, float],
        noisy_gates: Dict[str, float],
        qt: Optional[QubitTracking] = None,
        backend: Optional[BackendV2] = None,
        any_clifford_1: Optional[float] = None,
        any_clifford_2: Optional[float] = None,
        use_correlated_parity_measurement_errors: bool = False
    ):
        self.idle = idle
        self.measure_reset_idle = measure_reset_idle
        self.crosstalk_gates = crosstalk_gates
        self.noisy_gates = noisy_gates
        self.backend = backend
        self.qt = qt
        self.any_clifford_1 = any_clifford_1
        self.any_clifford_2 = any_clifford_2
        self.use_correlated_parity_measurement_errors = use_correlated_parity_measurement_errors

    @staticmethod
    def SD6(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=p,
            idle=p,
            measure_reset_idle=0,
            noisy_gates={
                "CX": p,
                "R": p,
                "M": p,
            },
        )

    @staticmethod
    def PC3(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=p,
            any_clifford_2=p,
            idle=p,
            measure_reset_idle=0,
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
            measure_reset_idle=0,
            any_clifford_1=p,
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
            any_clifford_1=0,
            any_clifford_2=0,
            idle=p,
            measure_reset_idle=0,
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
            any_clifford_1=p / 10,
            idle=p / 10,
            measure_reset_idle=2 * p,
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

    # TODO: FIX
    def add_crosstalk_errors_for_moment(self, touched_qubits: Set[int], post: stim.Circuit, gate: str):
        p = 0
        if gate in self.crosstalk_gates:
            p = self.crosstalk_gates[gate]
        elif gate in ANY_CLIFFORD_1_OPS and "SQ" in self.crosstalk_gates:
            p = self.crosstalk_gates["SQ"]
        elif gate in ANY_CLIFFORD_2_OPS and "TQ" in self.crosstalk_gates:
            p = self.crosstalk_gates["TQ"]
        
        already_noised = set()
        for q in touched_qubits:
            for neighbor in self.qt.get_neighbours(q):
                if neighbor in touched_qubits and neighbor not in already_noised:
                    if random.random() < p:
                        noise_op = "X_ERROR" if random.random() < 0.5 else "Z_ERROR"
                        post.append_operation(noise_op, [stim.target_qubit(neighbor)], 1.0)
                        already_noised.add(neighbor)

    def noisy_op(self, op: stim.CircuitInstruction, p: float, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if p > 0:
            if op.name in ANY_CLIFFORD_1_OPS:
                post.append_operation("DEPOLARIZE1", targets, p)
            # TODO: modified by adding SWAP_OPS
            elif op.name in ANY_CLIFFORD_2_OPS or op.name in SWAP_OPS:
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
        measured_or_reset_qubits: Set[int] = set()
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

                if op.name in self.noisy_gates:
                    p = self.noisy_gates[op.name]
                elif self.any_clifford_1 is not None and op.name in ANY_CLIFFORD_1_OPS:
                    p = self.any_clifford_1
                elif self.any_clifford_2 is not None and (op.name in ANY_CLIFFORD_2_OPS or op.name in SWAP_OPS):
                    p = self.any_clifford_2
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
                if op.name in MEASURE_OPS or op.name in RESET_OPS:
                    measured_or_reset_qubits |= touched_qubits
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
import stim
from utils.math import *
from typing import Dict, Any, Iterable, Callable, TypeVar, Tuple, Generic

from qiskit_qec.circuits import StimCodeCircuit

import dataclasses
import functools
from typing import List, FrozenSet, Optional

import stim


##########################################################################################
# https://github.com/Strilanc/heavy-hex-demo/
##########################################################################################
TItem = TypeVar('TItem')
T = TypeVar("T")

@dataclasses.dataclass(frozen=True)
class AtLayer(Generic[T]):
    """A special class that indicates the layer to read a measurement key from."""
    key: Any
    layer: int

def complex_key(c: complex) -> Any:
    return c.real != int(c.real), c.real, c.imag

def sorted_complex(
        values: Iterable[TItem],
        *,
        key: Callable[[TItem], Any] = lambda e: e) -> List[TItem]:
    return sorted(values, key=lambda e: complex_key(key(e)))


class MeasurementTracker:
    """Tracks measurements and groups of measurements, for producing stim record targets."""
    def __init__(self):
        self.recorded: Dict[Any, Optional[List[int]]] = {}
        self.next_measurement_index = 0

    def copy(self) -> 'MeasurementTracker':
        result = MeasurementTracker()
        result.recorded = {k: list(v) for k, v in self.recorded.items()}
        result.next_measurement_index = self.next_measurement_index
        return result

    def _rec(self, key: Any, value: Optional[List[int]]) -> None:
        if key in self.recorded:
            raise ValueError(f'Measurement key collision: {key=}')
        self.recorded[key] = value

    def record_measurement(self, key: Any) -> None:
        self._rec(key, [self.next_measurement_index])
        self.next_measurement_index += 1

    def make_measurement_group(self, sub_keys: Iterable[Any], *, key: Any) -> None:
        self._rec(key, self.measurement_indices(sub_keys))

    def record_obstacle(self, key: Any) -> None:
        self._rec(key, None)

    def measurement_indices(self, keys: Iterable[Any]) -> List[int]:
        result = set()
        for key in keys:
            if key not in self.recorded:
                raise ValueError(f"No such measurement: {key=}")
            for v in self.recorded[key]:
                if v is None:
                    raise ValueError(f"Obstacle at {key=}")
                if v in result:
                    result.remove(v)
                else:
                    result.add(v)
        return sorted(result)

    def current_measurement_record_targets_for(self, keys: Iterable[Any]) -> List[stim.GateTarget]:
        t0 = self.next_measurement_index
        times = self.measurement_indices(keys)
        return [stim.target_rec(t - t0) for t in sorted(times)]


class Builder:
    """Helper class for building stim circuits.

    Handles qubit indexing (complex -> int conversion).
    Handles measurement tracking (naming results and referring to them by name).
    """

    def __init__(self,
                 *,
                 q2i: Dict[complex, int],
                 circuit: stim.Circuit,
                 tracker: MeasurementTracker):
        self.q2i = q2i
        self.circuit = circuit
        self.tracker = tracker

    def copy(self) -> 'Builder':
        return Builder(q2i=dict(self.q2i), circuit=self.circuit.copy(), tracker=self.tracker.copy())

    @staticmethod
    def for_qubits(qubits: Iterable[complex]) -> 'Builder':
        q2i = {q: i for i, q in enumerate(sorted_complex(set(qubits)))}
        circuit = stim.Circuit()
        for q, i in q2i.items():
            circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])
        return Builder(
            q2i=q2i,
            circuit=circuit,
            tracker=MeasurementTracker(),
        )

    def gate(self,
             name: str,
             qubits: Iterable[complex]) -> None:
        qubits = sorted_complex(qubits)
        self.circuit.append(name, [self.q2i[q] for q in qubits])

    def shift_coords(self, *, dp: complex = 0, dt: int):
        self.circuit.append("SHIFT_COORDS", [], [dp.real, dp.imag, dt])

    def measure(self,
                qubits: Iterable[complex],
                *,
                basis: str = 'Z',
                tracker_key: Callable[[complex], Any] = lambda e: e,
                layer: int) -> None:
        qubits = sorted_complex(qubits)
        self.circuit.append(f"M{basis}", [self.q2i[q] for q in qubits])
        for q in qubits:
            self.tracker.record_measurement(AtLayer(tracker_key(q), layer))

    def measure_pauli_product(self,
                              *,
                              xs: Iterable[complex] = (),
                              ys: Iterable[complex] = (),
                              zs: Iterable[complex] = (),
                              qs: Dict[str, Iterable[complex]] = None,
                              key: Any,
                              layer: int = -1):
        x = set(xs)
        y = set(ys)
        z = set(zs)
        if qs is not None:
            for b, bqs in qs.items():
                if b == 'X':
                    x |= set(bqs)
                elif b == 'Y':
                    y |= set(bqs)
                elif b == 'Z':
                    z |= set(bqs)
                else:
                    raise NotImplementedError(f'{b=}')
        xz = x & z
        xy = x & y
        yz = y & z
        x -= xz
        x -= xy
        z -= xz
        z -= yz
        y -= xy
        y -= yz
        x |= yz
        y |= xz
        z |= xy
        vals = {}
        for q in x:
            vals[q] = stim.target_x(self.q2i[q])
        for q in y:
            vals[q] = stim.target_y(self.q2i[q])
        for q in z:
            vals[q] = stim.target_z(self.q2i[q])

        targets = []
        comb = stim.target_combiner()
        for q in sorted_complex(vals.keys()):
            targets.append(vals[q])
            targets.append(comb)
        if targets:
            targets.pop()
            self.circuit.append('MPP', targets)
            self.tracker.record_measurement(AtLayer(key, layer))
        else:
            self.tracker.make_measurement_group([], key=AtLayer(key, layer))

    def detector(self,
                 keys: Iterable[Any],
                 *,
                 pos: Optional[complex] = None,
                 t: int = 0,
                 mark_as_post_selected: bool = False,
                 ignore_non_existent: bool = False) -> None:
        if pos is not None:
            coords = [pos.real, pos.imag, t]
            if mark_as_post_selected:
                coords.append(1)
        else:
            if mark_as_post_selected:
                raise ValueError('pos is None and mark_as_post_selected')
            coords = None

        if ignore_non_existent:
            keys = [k for k in keys if k in self.tracker.recorded]
        targets = self.tracker.current_measurement_record_targets_for(keys)
        self.circuit.append('DETECTOR', targets, coords)

    def obs_include(self,
                    keys: Iterable[Any],
                    *,
                    obs_index: int) -> None:
        self.circuit.append(
            'OBSERVABLE_INCLUDE',
            self.tracker.current_measurement_record_targets_for(keys),
            obs_index,
        )

    def tick(self) -> None:
        self.circuit.append('TICK')

    def cx(self, pairs: List[Tuple[complex, complex]]) -> None:
        sorted_pairs = []
        for a, b in pairs:
            sorted_pairs.append((a, b))
        sorted_pairs = sorted(sorted_pairs, key=lambda e: (complex_key(e[0]), complex_key(e[1])))
        for a, b in sorted_pairs:
            self.circuit.append('CX', [self.q2i[a], self.q2i[b]])

    def cz(self, pairs: List[Tuple[complex, complex]]) -> None:
        sorted_pairs = []
        for a, b in pairs:
            if complex_key(a) > complex_key(b):
                a, b = b, a
            sorted_pairs.append((a, b))
        sorted_pairs = sorted(sorted_pairs, key=lambda e: (complex_key(e[0]), complex_key(e[1])))
        for a, b in sorted_pairs:
            self.circuit.append('CZ', [self.q2i[a], self.q2i[b]])

    def classical_paulis(self,
                         *,
                         control_keys: Iterable[Any],
                         targets: Iterable[complex],
                         basis: str) -> None:
        gate = f'C{basis}'
        indices = [self.q2i[q] for q in sorted_complex(targets)]
        for rec in self.tracker.current_measurement_record_targets_for(control_keys):
            for i in indices:
                self.circuit.append(gate, [rec, i])

@dataclasses.dataclass
class Tile:
    data_qubits: List[Optional[complex]]
    measure_qubit: complex
    basis: str

    @functools.cached_property
    def m(self) -> complex:
        return self.measure_qubit

    @functools.cached_property
    def degree(self) -> int:
        return len(self.data_set)

    @functools.cached_property
    def data_set(self) -> FrozenSet[complex]:
        return frozenset(q
                         for q in self.data_qubits
                         if q is not None)

    @functools.cached_property
    def used_set(self) -> FrozenSet[complex]:
        return self.data_set | frozenset([self.measure_qubit])


def checkerboard_basis(c: complex) -> str:
    return 'Z' if (c.real + c.imag) % 2 == 1 else 'X'


def create_heavy_hex_tiles(diam: int) -> List[Tile]:
    all_data_set = {
        x + 1j*y
        for x in range(diam)
        for y in range(diam)
    }
    tiles = []
    top_basis = 'X'
    side_basis = 'Z'
    for x in range(-1, diam):
        for y in range(-1, diam):
            top_left = x + 1j*y
            center = top_left + 0.5 + 0.5j
            b = checkerboard_basis(center)
            on_top_or_bottom = y in [-1, diam - 1]
            on_side = x in [-1, diam - 1]
            if on_top_or_bottom and b != top_basis:
                continue
            if on_side and b != side_basis:
                continue

            data_qubits = [
                top_left,
                top_left + 1,
                top_left + 1j,
                top_left + 1 + 1j,
            ]
            if b == 'Z':
                continue
            for qi in range(4):
                if data_qubits[qi] not in all_data_set:
                    data_qubits[qi] = None
            degree = sum(e is not None for e in data_qubits)
            if degree < 2:
                continue
            rem_center = sum(q for q in data_qubits if q is not None) / degree
            tiles.append(Tile(
                data_qubits=data_qubits,
                measure_qubit=rem_center,
                basis=b,
            ))
    for x in range(diam):
        for y in range(diam - 1):
            q = x + y*1j
            tiles.append(Tile(
                data_qubits=[q, q + 1j],
                measure_qubit=q + 0.5j,
                basis='Z',
            ))
    return tiles


def make_mpp_based_round(*,
                         layer: int,
                         tiles: List[Tile],
                         builder: Builder,
                         time_boundary_basis: str,
                         x_combos: List[List[Tile]],
                         z_combos: List[List[Optional[Tile]]]):
    for desired_parity in [False, True]:
        for tile in tiles:
            parity = tile.measure_qubit.real % 2 == 0.5
            if tile.basis == 'X' and parity == desired_parity:
                builder.measure_pauli_product(
                    qs={tile.basis: tile.data_set},
                    key=tile.measure_qubit,
                    layer=layer,
                )
        builder.tick()

    for desired_parity in [False, True]:
        for tile in tiles:
            parity = tile.measure_qubit.imag % 2 == 0.5
            if tile.basis == 'Z' and parity == desired_parity:
                builder.measure_pauli_product(
                    qs={tile.basis: tile.data_set},
                    key=tile.measure_qubit,
                    layer=layer,
                )
        if not desired_parity:
            builder.tick()

    # Combined X column detectors
    if layer > 0 or time_boundary_basis == 'X':
        for combined_tiles in x_combos:
            pos = sum(tile.measure_qubit for tile in combined_tiles) / len(combined_tiles)
            pos = pos.real - 10j
            builder.detector([
                AtLayer(tile.measure_qubit, layer - d)
                for tile in combined_tiles
                for d in ([0, 1] if layer > 0 else [0])
            ], pos=pos)

    # Z detectors
    if layer > 0 or time_boundary_basis == 'Z':
        for combined_tiles in z_combos:
            kept_tiles = [tile for tile in combined_tiles if tile is not None]
            builder.detector([
                AtLayer(tile.measure_qubit, layer - d)
                for tile in kept_tiles
                for d in ([0, 1] if layer > 0 else [0])
            ], pos=sum(tile.measure_qubit for tile in kept_tiles) / len(kept_tiles))
    builder.shift_coords(dt=1)
    builder.tick()


def make_cx_based_round(*,
                        layer: int,
                        tiles: List[Tile],
                        builder: Builder,
                        time_boundary_basis: str,
                        x_combos: List[List[Tile]],
                        z_combos: List[List[Optional[Tile]]],
                        include_flags: bool):
    # step = 1
    builder.gate("R", [t.m for t in tiles])
    builder.tick()
    builder.gate("H", [t.m for t in tiles if t.basis == 'X'])
    builder.tick()
    # step = 2
    builder.cx([
        (t.m, t.m + 0.5)
        for t in tiles
        if t.degree == 4
    ])
    builder.tick()
    # step = 3
    builder.cx([
        (t.m + 0.5, t.m + 0.5 - 0.5j)
        for t in tiles
        if t.degree == 4
    ])
    builder.cx([
        (t.m, t.m - 0.5)
        for t in tiles
        if t.degree == 4
    ])
    builder.tick()
    # step = 4
    builder.cx([
        (t.m + 0.5, t.m + 0.5 + 0.5j)
        for t in tiles
        if t.degree == 4
    ])
    builder.cx([
        (t.m - 0.5, t.m - 0.5 + 0.5j)
        for t in tiles
        if t.degree == 4
    ])
    builder.cx([
        (t.m, t.m + 0.5)
        for t in tiles
        if t.basis == 'X'
        if t.degree == 2
        if t.data_qubits[0] is None  # (on top of patch)
    ])
    builder.tick()
    # step = 5
    builder.cx([
        (t.m - 0.5, t.m - 0.5 - 0.5j)
        for t in tiles
        if t.degree == 4
    ])
    builder.cx([
        (t.m, t.m + 0.5)
        for t in tiles
        if t.degree == 4
    ])
    builder.cx([
        (t.m, t.m - 0.5)
        for t in tiles
        if t.basis == 'X'
        if t.degree == 2
        if t.data_qubits[0] is None  # (on top of patch)
    ])
    builder.cx([
        (t.m, t.m + 0.5)
        for t in tiles
        if t.basis == 'X'
        if t.degree == 2
        if t.data_qubits[0] is not None  # (on bottom of patch)
    ])
    builder.tick()
    # step = 6
    builder.cx([
        (t.m, t.m - 0.5)
        for t in tiles
        if t.degree == 4
    ])
    builder.cx([
        (t.m, t.m - 0.5)
        for t in tiles
        if t.basis == 'X'
        if t.degree == 2
        if t.data_qubits[0] is not None  # (on bottom of patch)
    ])
    builder.tick()
    # step = 7
    builder.gate("H", [t.m for t in tiles if t.basis == 'X'])
    builder.tick()
    builder.measure([t.m for t in tiles if t.basis == 'X'], layer=layer)
    flags = [
        t.m + d
        for t in tiles
        if t.basis == 'X' and t.degree == 4
        for d in [0.5, -0.5]
    ]
    if include_flags:
        builder.measure(flags,
                        layer=layer,
                        tracker_key=lambda c: ('flag', c))
        for f in flags:
            builder.detector([AtLayer(('flag', f), layer)], pos=f + 0.25 + 0.25j)
    builder.shift_coords(dt=1)
    builder.tick()
    builder.gate('R', [t.m for t in tiles if t.basis == 'Z'])
    builder.tick()
    # step = 8
    builder.cx([
        (left.m - 0.5j, left.m)
        for left, right in z_combos
        if left is not None
    ])
    builder.tick()
    # step = 9
    builder.cx([
        (left.m + 0.5j, left.m)
        for left, right in z_combos
        if left is not None
    ])
    builder.cx([
        (right.m + 0.5j, right.m)
        for left, right in z_combos
        if right is not None
    ])
    builder.tick()
    # step = 10
    builder.cx([
        (right.m - 0.5j, right.m)
        for left, right in z_combos
        if right is not None
    ])
    builder.tick()
    # step = 11
    builder.measure([t.m for t in tiles if t.basis == 'Z'], layer=layer)

    # Combined X column detectors
    if layer > 0 or time_boundary_basis == 'X':
        for combined_tiles in x_combos:
            pos = sum(tile.measure_qubit for tile in combined_tiles) / len(combined_tiles)
            pos = pos.real - 10j
            builder.detector([
                AtLayer(tile.measure_qubit, layer - d)
                for tile in combined_tiles
                for d in ([0, 1] if layer > 0 else [0])
            ], pos=pos)

    # Z detectors
    if layer > 0 or time_boundary_basis == 'Z':
        for combined_tiles in z_combos:
            kept_tiles = [tile for tile in combined_tiles if tile is not None]
            pos = sum(tile.measure_qubit for tile in kept_tiles) / len(kept_tiles)
            builder.detector([
                AtLayer(tile.measure_qubit, layer - d)
                for tile in kept_tiles
                for d in ([0, 1] if layer > 0 else [0])
            ], pos=pos)
    builder.shift_coords(dt=1)
    builder.tick()


def make_heavy_hex_circuit(
        *,
        diam: int,
        time_boundary_basis: str,
        rounds: int,
        gate_set: str,
) -> stim.Circuit:
    tiles = create_heavy_hex_tiles(diam)
    data_set = {q for tile in tiles for q in tile.data_set}
    used_set = {q for tile in tiles for q in tile.used_set}
    m2t = {tile.measure_qubit: tile for tile in tiles}
    for tile in tiles:
        if tile.basis == 'X' and tile.degree == 2:
            if tile.data_qubits[0] is None:  # top of patch
                m2t[tile.m - 0.5j] = tile
            else:
                m2t[tile.m + 0.5j] = tile

    x_combos: List[List[Tile]] = []
    for col in range(diam - 1):
        combined_tiles = []
        for row in range(-1, diam + 1):
            m = col + row * 1j + 0.5 + 0.5j
            if m in m2t:
                assert row != diam
                combined_tiles.append(m2t[m])
                assert m2t[m].basis == 'X'
        x_combos.append(combined_tiles)
    z_combos: List[List[Optional[Tile]]] = []
    for x in range(-1, diam):
        for y in range(diam - 1):
            m = x + y * 1j + 0.5 + 0.5j
            if checkerboard_basis(m) == 'X':
                continue
            combined_tiles = []
            for m2 in [m - 0.5, m + 0.5]:
                if m2 in m2t:
                    combined_tiles.append(m2t[m2])
                else:
                    combined_tiles.append(None)
            if combined_tiles:
                z_combos.append(combined_tiles)

    builder = Builder.for_qubits(used_set)
    builder.gate("R", data_set)
    builder.tick()
    if time_boundary_basis == 'X':
        builder.gate("H", data_set)
        builder.tick()

    if gate_set == 'mpp':
        round_maker = make_mpp_based_round
    elif gate_set == 'cx':
        round_maker = functools.partial(make_cx_based_round, include_flags=True)
    elif gate_set == 'cx_noflags':
        round_maker = functools.partial(make_cx_based_round, include_flags=False)
    else:
        raise NotImplementedError(f'{gate_set=}')

    assert rounds >= 2
    round_maker(layer=0,
                tiles=tiles,
                builder=builder,
                time_boundary_basis=time_boundary_basis,
                x_combos=x_combos,
                z_combos=z_combos)

    head = builder.circuit.copy()
    builder.circuit.clear()
    layer = 1
    round_maker(layer=1,
                tiles=tiles,
                builder=builder,
                time_boundary_basis=time_boundary_basis,
                x_combos=x_combos,
                z_combos=z_combos)

    if rounds > 2:
        builder.circuit *= (rounds - 2)
        layer += 1
        round_maker(layer=layer,
                    tiles=tiles,
                    builder=builder,
                    time_boundary_basis=time_boundary_basis,
                    x_combos=x_combos,
                    z_combos=z_combos)

    if time_boundary_basis == 'X':
        builder.gate('H', data_set)
        builder.tick()
    builder.measure(data_set, layer=layer)

    # Final detectors
    final_tiles = x_combos if time_boundary_basis == 'X' else z_combos
    for combined_tiles in final_tiles:
        kept_tiles = [tile for tile in combined_tiles if tile is not None]
        pos = sum(tile.measure_qubit for tile in kept_tiles) / len(kept_tiles)
        if time_boundary_basis == 'X':
            pos = pos.real - 10j
        builder.detector({
            AtLayer(q, layer)
            for tile in kept_tiles
            for q in tile.used_set
        }, pos=pos)

    if time_boundary_basis == 'X':
        obs_qubits = {q for q in data_set if q.real == 0}
    else:
        obs_qubits = {q for q in data_set if q.imag == 0}
    builder.obs_include([AtLayer(q, layer) for q in obs_qubits],
                        obs_index=0)
    return head + builder.circuit



##########################################################################################


def get_hh_code(d, T = None):
    if T == None:
        T = d
    stim_circuit = make_heavy_hex_circuit(
        diam=d,
        time_boundary_basis='Z',
        rounds=T,
        gate_set='cx',
    )
    return StimCodeCircuit(stim_circuit = stim_circuit)
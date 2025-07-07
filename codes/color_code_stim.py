import math
from functools import wraps
from typing import (
    Iterable,
    Optional,
    Union,
)

import igraph as ig
import numpy as np
import stim

from qiskit_qec.circuits import StimCodeCircuit


#########################################################################
# Functions adapted from https://github.com/seokhyung-lee/color-code-stim
#########################################################################

class ColorCode:
    tanner_graph: ig.Graph
    circuit: stim.Circuit
    d: int
    rounds: int
    qubit_groups: dict
    org_dem: stim.DetectorErrorModel
    _bp_inputs: dict

    def __init__(
        self,
        *,
        d: int,
        rounds: int,
        shape: str = "tri",
        d2: int = None,
        cnot_schedule: Union[str, Iterable[int]] = "tri_optimal",
        p_bitflip: float = 0.0,
        p_reset: float = 0.0,
        p_meas: float = 0.0,
        p_cnot: float = 0.0,
        p_idle: float = 0.0,
        p_circuit: Optional[float] = None,
        perfect_init_final: bool = False,
        logical_gap: bool = False,
        use_last_detectors: bool = True,
        # custom_noise_channel: Optional[Tuple[str, object]] = None,
        # dem_decomposed: Optional[Dict[str, Tuple[
        #     stim.DetectorErrorModel, stim.DetectorErrorModel]]] =
        #     None,
        benchmarking: bool = False,
    ):
        """
        Class for constructing a color code circuit and simulating the
        concatenated MWPM decoder.

        Parameters
        ----------
        d : int >= 3
            Code distance. Should be an odd number of 3 or more.
        rounds : int >= 1
            Number of syndrome extraction rounds.
        shape : {'triangle', 'tri', 'rectangle', 'rec', 'rec_stability', 'growing'}, default 'tri'
            Shape of the color code patch.
        d2 : int >= 3, optional
            Second code distance of the rectangular patch (if applicable). If
            not provided, d2=d.
        cnot_schedule : {12-tuple of integers, 'tri_optimal', 'tri_optimal_reversed'}, default 'tri_optimal'
            CNOT schedule.
            If this is a 12-tuple of integers, it indicate (a, b, ... l)
            specifying the CNOT schedule.
            If this is 'tri_optimal', it is (2, 3, 6, 5, 4, 1, 3, 4, 7, 6,
            5, 2), which is the optimal schedule for the triangular color code.
            If this is 'tri_optimal_reversed', it is (3, 4, 7, 6, 5, 2, 2,
            3, 6, 5, 4, 1), which has the X- and Z-part reversed from
            'tri_optimal'.
        p_bitflip : float, default 0
            Bit-flip probability at the start of each round.
        p_reset : float, default 0
            Probability of a wrong qubit reset (i.e., producing an
            orthogonal state).
        p_meas : float, default 0
            Probability of a flipped measurement outcome.
        p_cnot : float, default 0
            Strength of a two-qubit depolarizing noise channel following
            each CNOT gate.
        p_idle : float, default 0
            Strength of a single-qubit depolarizing noise channel following
            each idle gate.
        p_circuit : float, optional
            If given, p_reset = p_meas = p_cnot = p_idle = p_circuit.
        perfect_init_final : bool, default False
            Whether to use perfect initialization and final measurement.
        logical_gap : bool, default False
            Whether to compute logical gap during decoding.
        use_last_detectors : bool, default True
            Whether to use detectors from the last round.
        benchmarking : bool, default False
            Whether to measure execution time of each step.
        """
        if isinstance(cnot_schedule, str):
            if cnot_schedule in ["tri_optimal", "LLB"]:
                cnot_schedule = (2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2)
            elif cnot_schedule == "tri_optimal_reversed":
                cnot_schedule = (3, 4, 7, 6, 5, 2, 2, 3, 6, 5, 4, 1)
            else:
                raise ValueError
        else:
            assert len(cnot_schedule) == 12

        assert d > 1 and rounds >= 1

        if p_circuit is not None:
            p_reset = p_meas = p_cnot = p_idle = p_circuit

        self.d = d
        d2 = self.d2 = d if d2 is None else d2
        self.rounds = rounds
        if shape in {"triangle", "tri"}:
            self.shape = "tri"
            self.num_obs = 1
            assert d % 2 == 1

        elif shape in {"rectangle", "rec"}:
            self.shape = "rec"
            self.num_obs = 2
            assert d % 2 == 0 and d2 % 2 == 0

        elif shape == "rec_stability":
            self.shape = "rec_stability"
            self.num_obs = 2
            assert d % 2 == 0 and d2 % 2 == 0

        elif shape == "growing":
            self.shape = "growing"
            self.num_obs = 1
            assert d % 2 == 1 and d2 % 2 == 1 and d2 > d

        else:
            raise ValueError("Invalid shape")

        self.cnot_schedule = cnot_schedule
        self.perfect_init_final = perfect_init_final
        self.probs = {
            "bitflip": p_bitflip,
            "reset": p_reset,
            "meas": p_meas,
            "cnot": p_cnot,
            "idle": p_idle,
        }
        self.logical_gap = logical_gap
        if logical_gap and self.shape not in {"tri", "growing"}:
            raise NotImplementedError
        self.use_last_detectors = use_last_detectors

        self.tanner_graph = ig.Graph()
        self.circuit = stim.Circuit()

        self.benchmarking = benchmarking

        # Mapping between detector ids and ancillary qubits
        # self.detectors[detector_id] = (anc_qubit, time_coord)
        self.detectors = []

        # Detector ids grouped by colors
        self.detector_ids = {"r": [], "g": [], "b": []}

        # Various qubit groups
        self.qubit_groups = {}

        # Decomposed detector error models
        # It is generated when required.
        # if dem_decomposed is None:
        #     dem_decomposed = {}
        self.dems_sym_decomposed = {}
        self.dems_decomposed = {}

        tanner_graph = self.tanner_graph

        self._create_tanner_graph()

        self._generate_circuit()

        self.org_dem = self.circuit.detector_error_model(flatten_loops=True)
        self._bp_inputs = {}

        # Get detector list
        detector_coords_dict = self.circuit.get_detector_coordinates()
        for detector_id in range(self.circuit.num_detectors):
            coords = detector_coords_dict[detector_id]
            if len(coords) == 1:
                # boundary dets when logical_gap = True
                if self.shape == "tri":
                    color = "r"
                elif self.shape == "growing":
                    color = "g"
                else:
                    raise NotImplementedError
            else:
                x = math.floor(coords[0])
                y = round(coords[1])
                t = round(coords[2])
                try:
                    name = f"{x}-{y}-X"
                    qubit = tanner_graph.vs.find(name=name)
                except ValueError:
                    name = f"{x + 1}-{y}-Z"
                    qubit = tanner_graph.vs.find(name=name)
                self.detectors.append((qubit, t))
                color = qubit["color"]

            self.detector_ids[color].append(detector_id)

    def _create_tanner_graph(self):
        shape = self.shape
        tanner_graph = self.tanner_graph

        if shape in {"tri", "growing"}:
            if shape == "tri":
                d = self.d
            else:
                d = self.d2

            assert d % 2 == 1

            detid = 0
            L = round(3 * (d - 1) / 2)
            for y in range(L + 1):
                if y % 3 == 0:
                    anc_qubit_color = "g"
                    anc_qubit_pos = 2
                elif y % 3 == 1:
                    anc_qubit_color = "b"
                    anc_qubit_pos = 0
                else:
                    anc_qubit_color = "r"
                    anc_qubit_pos = 1

                for x in range(y, 2 * L - y + 1, 2):
                    boundary = []
                    if y == 0:
                        boundary.append("r")
                    if x == y:
                        boundary.append("g")
                    if x == 2 * L - y:
                        boundary.append("b")
                    boundary = "".join(boundary)
                    if not boundary:
                        boundary = None

                    if shape == "tri":
                        obs = boundary in ["r", "rg", "rb"]
                    elif shape == "growing":
                        obs = boundary in ["g", "gb", "rg"]
                    else:
                        obs = False

                    if round((x - y) / 2) % 3 != anc_qubit_pos:
                        tanner_graph.add_vertex(
                            name=f"{x}-{y}",
                            x=x,
                            y=y,
                            qid=tanner_graph.vcount(),
                            pauli=None,
                            color=None,
                            obs=obs,
                            boundary=boundary,
                        )
                    else:
                        for pauli in ["Z", "X"]:
                            tanner_graph.add_vertex(
                                name=f"{x}-{y}-{pauli}",
                                x=x,
                                y=y,
                                qid=tanner_graph.vcount(),
                                pauli=pauli,
                                color=anc_qubit_color,
                                obs=False,
                                boundary=boundary,
                            )
                            detid += 1

        elif shape == "rec":
            d, d2 = self.d, self.d2
            assert d % 2 == 0
            assert d2 % 2 == 0

            detid = 0
            L1 = round(3 * d / 2 - 2)
            L2 = round(3 * d2 / 2 - 2)
            for y in range(L2 + 1):
                if y % 3 == 0:
                    anc_qubit_color = "g"
                    anc_qubit_pos = 2
                elif y % 3 == 1:
                    anc_qubit_color = "b"
                    anc_qubit_pos = 0
                else:
                    anc_qubit_color = "r"
                    anc_qubit_pos = 1

                for x in range(y, y + 2 * L1 + 1, 2):
                    boundary = []
                    if y == 0 or y == L2:
                        boundary.append("r")
                    if y == x or y == x - 2 * L1:
                        boundary.append("g")
                    boundary = "".join(boundary)
                    if not boundary:
                        boundary = None

                    if round((x - y) / 2) % 3 != anc_qubit_pos:
                        obs_g = y == 0
                        obs_r = x == y + 2 * L1

                        tanner_graph.add_vertex(
                            name=f"{x}-{y}",
                            x=x,
                            y=y,
                            qid=tanner_graph.vcount(),
                            pauli=None,
                            color=None,
                            obs_r=obs_r,
                            obs_g=obs_g,
                            boundary=boundary,
                        )
                    else:
                        for pauli in ["Z", "X"]:
                            tanner_graph.add_vertex(
                                name=f"{x}-{y}-{pauli}",
                                x=x,
                                y=y,
                                qid=tanner_graph.vcount(),
                                pauli=pauli,
                                color=anc_qubit_color,
                                obs_r=False,
                                obs_g=False,
                                boundary=boundary,
                            )
                            detid += 1

            # Additional corner vertex
            x = y = L2 + 1
            # x, y = L2 - 2, L2
            tanner_graph.add_vertex(
                name=f"{x}-{y}",
                x=x,
                y=y,
                qid=tanner_graph.vcount(),
                pauli=None,
                color=None,
                obs_r=False,
                obs_g=False,
                boundary="rg",
            )

        elif shape == "rec_stability":
            d = self.d
            d2 = self.d2
            assert d % 2 == 0
            assert d2 % 2 == 0

            detid = 0
            L1 = round(3 * d / 2 - 2)
            L2 = round(3 * d2 / 2 - 2)
            for y in range(L2 + 1):
                if y % 3 == 0:
                    anc_qubit_color = "r"
                    anc_qubit_pos = 0
                elif y % 3 == 1:
                    anc_qubit_color = "b"
                    anc_qubit_pos = 1
                else:
                    anc_qubit_color = "g"
                    anc_qubit_pos = 2

                if y == 0:
                    x_init_adj = 4
                elif y == 1:
                    x_init_adj = 2
                else:
                    x_init_adj = 0

                if y == L2:
                    x_fin_adj = 4
                elif y == L2 - 1:
                    x_fin_adj = 2
                else:
                    x_fin_adj = 0

                for x in range(y + x_init_adj, y + 2 * L1 + 1 - x_fin_adj, 2):
                    if (
                        y == 0
                        or y == L2
                        or x == y
                        or x == y + 2 * L1
                        or (x, y) == (3, 1)
                        or (x, y) == (L2 + 2 * L1 - 3, L2 - 1)
                    ):
                        boundary = "g"
                    else:
                        boundary = None

                    if round((x - y) / 2) % 3 != anc_qubit_pos:
                        tanner_graph.add_vertex(
                            name=f"{x}-{y}",
                            x=x,
                            y=y,
                            qid=tanner_graph.vcount(),
                            pauli=None,
                            color=None,
                            boundary=boundary,
                        )
                    else:
                        for pauli in ["Z", "X"]:
                            tanner_graph.add_vertex(
                                name=f"{x}-{y}-{pauli}",
                                x=x,
                                y=y,
                                qid=tanner_graph.vcount(),
                                pauli=pauli,
                                color=anc_qubit_color,
                                boundary=boundary,
                            )
                            detid += 1

        else:
            raise ValueError("Invalid shape")

        # Update qubit_groups
        data_qubits = tanner_graph.vs.select(pauli=None)
        anc_qubits = tanner_graph.vs.select(pauli_ne=None)
        anc_Z_qubits = anc_qubits.select(pauli="Z")
        anc_X_qubits = anc_qubits.select(pauli="X")
        anc_red_qubits = anc_qubits.select(color="r")
        anc_green_qubits = anc_qubits.select(color="g")
        anc_blue_qubits = anc_qubits.select(color="b")

        self.qubit_groups.update(
            {
                "data": data_qubits,
                "anc": anc_qubits,
                "anc_Z": anc_Z_qubits,
                "anc_X": anc_X_qubits,
                "anc_red": anc_red_qubits,
                "anc_green": anc_green_qubits,
                "anc_blue": anc_blue_qubits,
            }
        )

        # Add edges
        links = []
        offsets = [(-1, 1), (1, 1), (2, 0), (1, -1), (-1, -1), (-2, 0)]
        for anc_qubit in self.qubit_groups["anc"]:
            data_qubits = []
            for offset in offsets:
                data_qubit_x = anc_qubit["x"] + offset[0]
                data_qubit_y = anc_qubit["y"] + offset[1]
                data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
                try:
                    data_qubit = tanner_graph.vs.find(name=data_qubit_name)
                except ValueError:
                    continue
                data_qubits.append(data_qubit)
                tanner_graph.add_edge(anc_qubit, data_qubit, kind="tanner", color=None)

            if anc_qubit["pauli"] == "Z":
                weight = len(data_qubits)
                for i in range(weight):
                    qubit = data_qubits[i]
                    next_qubit = data_qubits[(i + 1) % weight]
                    if not tanner_graph.are_connected(qubit, next_qubit):
                        link = tanner_graph.add_edge(
                            qubit, next_qubit, kind="lattice", color=None
                        )
                        links.append(link)

        # Assign colors to links
        link: ig.Edge
        for link in links:
            v1: ig.Vertex
            v2: ig.Vertex
            v1, v2 = link.target_vertex, link.source_vertex
            ngh_ancs_1 = {anc.index for anc in v1.neighbors() if anc["pauli"] == "Z"}
            ngh_ancs_2 = {anc.index for anc in v2.neighbors() if anc["pauli"] == "Z"}
            color = tanner_graph.vs[(ngh_ancs_1 ^ ngh_ancs_2).pop()]["color"]
            link["color"] = color

    def _generate_circuit(self):
        qubit_groups = self.qubit_groups
        cnot_schedule = self.cnot_schedule
        tanner_graph = self.tanner_graph
        circuit = self.circuit
        rounds = self.rounds
        shape = self.shape
        d = self.d
        d2 = self.d2

        probs = self.probs
        p_bitflip = probs["bitflip"]
        p_reset = probs["reset"]
        p_meas = probs["meas"]
        p_cnot = probs["cnot"]
        p_idle = probs["idle"]

        perfect_init_final = self.perfect_init_final
        use_last_detectors = self.use_last_detectors

        data_qubits = qubit_groups["data"]
        anc_qubits = qubit_groups["anc"]
        anc_Z_qubits = qubit_groups["anc_Z"]
        anc_X_qubits = qubit_groups["anc_X"]

        data_qids = data_qubits["qid"]
        anc_qids = anc_qubits["qid"]
        anc_Z_qids = anc_Z_qubits["qid"]
        anc_X_qids = anc_X_qubits["qid"]

        num_data_qubits = len(data_qids)
        num_anc_Z_qubits = len(anc_Z_qubits)
        num_anc_X_qubits = len(anc_X_qubits)
        num_anc_qubits = num_anc_X_qubits + num_anc_Z_qubits

        num_qubits = tanner_graph.vcount()
        all_qids = list(range(num_qubits))
        all_qids_set = set(all_qids)

        if shape == "rec_stability":
            temp_bdrys = "r"
            red_links = [
                [link.source, link.target] for link in tanner_graph.es.select(color="r")
            ]
            red_links = np.array(red_links)
            data_q1s = red_links[:, 0]
            data_q2s = red_links[:, 1]
        elif shape in {"tri", "rec"}:
            temp_bdrys = "z"
            red_links = data_q1s = data_q2s = None
        elif shape == "growing":
            temp_bdrys = "mixed"  # Mixed boundaries
            y_init_patch_bdry = 3 * round((d2 - d) / 2)
            data_qubits_outside_init_patch = data_qubits.select(y_lt=y_init_patch_bdry)
            red_links = [
                [link.source, link.target]
                for link in tanner_graph.es.select(
                    color="r",
                    _within=data_qubits_outside_init_patch,
                )
            ]
            red_links = np.array(red_links)
            data_q1s = red_links[:, 0]
            data_q2s = red_links[:, 1]
        else:
            raise NotImplementedError

        # custom_noise_channel = self.custom_noise_channel

        # Syndrome extraction circuit without SPAM
        synd_extr_circuit_without_spam = stim.Circuit()
        for timeslice in range(1, max(cnot_schedule) + 1):
            targets = [i for i, val in enumerate(cnot_schedule) if val == timeslice]
            operated_qids = set()

            CX_targets = []
            for target in targets:
                if target in {0, 6}:
                    offset = (-1, 1)
                elif target in {1, 7}:
                    offset = (1, 1)
                elif target in {2, 8}:
                    offset = (2, 0)
                elif target in {3, 9}:
                    offset = (1, -1)
                elif target in {4, 10}:
                    offset = (-1, -1)
                else:
                    offset = (-2, 0)

                target_anc_qubits = anc_Z_qubits if target < 6 else anc_X_qubits
                for anc_qubit in target_anc_qubits:
                    data_qubit_x = anc_qubit["x"] + offset[0]
                    data_qubit_y = anc_qubit["y"] + offset[1]
                    data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
                    try:
                        data_qubit = tanner_graph.vs.find(name=data_qubit_name)
                    except ValueError:
                        continue
                    anc_qid = anc_qubit.index
                    data_qid = data_qubit.index
                    operated_qids.update({anc_qid, data_qid})

                    # tanner_graph.add_edge(anc_qid, data_qid)
                    CX_target = (
                        [data_qid, anc_qid] if target < 6 else [anc_qid, data_qid]
                    )
                    CX_targets.extend(CX_target)

            synd_extr_circuit_without_spam.append("CX", CX_targets)
            if p_cnot > 0:
                synd_extr_circuit_without_spam.append("DEPOLARIZE2", CX_targets, p_cnot)

            if p_idle > 0:
                idling_qids = list(all_qids_set - operated_qids)
                synd_extr_circuit_without_spam.append(
                    "DEPOLARIZE1", idling_qids, p_idle
                )

            synd_extr_circuit_without_spam.append("TICK")

        def get_qubit_coords(qubit: ig.Vertex):
            coords = [qubit["x"], qubit["y"]]
            if qubit["pauli"] == "Z":
                coords[0] -= 0.5
            elif qubit["pauli"] == "X":
                coords[0] += 0.5

            return tuple(coords)

        # Syndrome extraction circuit with measurement & detector
        def get_synd_extr_circuit(first=False):
            synd_extr_circuit = synd_extr_circuit_without_spam.copy()

            synd_extr_circuit.append("MRZ", anc_Z_qids, p_meas)

            if first:
                for j, anc_qubit in enumerate(anc_Z_qubits):
                    if anc_qubit["color"] == "r":
                        if temp_bdrys == "r":
                            continue
                        if (
                            temp_bdrys == "mixed"
                            and anc_qubit["y"] < y_init_patch_bdry - 0.1
                        ):
                            continue
                    lookback = -num_anc_Z_qubits + j
                    coords = get_qubit_coords(anc_qubit)
                    coords += (0,)
                    target = stim.target_rec(lookback)
                    synd_extr_circuit.append("DETECTOR", target, coords)

            else:
                for j, anc_qubit in enumerate(anc_Z_qubits):
                    lookback = -num_anc_Z_qubits + j
                    coords = get_qubit_coords(anc_qubit)
                    coords += (0,)
                    target = [
                        stim.target_rec(lookback),
                        stim.target_rec(lookback - num_anc_qubits),
                    ]
                    synd_extr_circuit.append("DETECTOR", target, coords)

            synd_extr_circuit.append("MRX", anc_X_qids, p_meas)
            if first and temp_bdrys in {"r", "mixed"}:
                for j, anc_qubit in enumerate(anc_X_qubits):
                    if temp_bdrys == "r" and anc_qubit["color"] == "r":
                        continue
                    elif temp_bdrys == "mixed" and (
                        anc_qubit["color"] == "r" or anc_qubit["y"] >= y_init_patch_bdry
                    ):
                        continue

                    lookback = -num_anc_X_qubits + j
                    coords = get_qubit_coords(anc_qubit)
                    coords += (0,)
                    target = [stim.target_rec(lookback)]
                    synd_extr_circuit.append("DETECTOR", target, coords)

            elif not first:
                for j, anc_qubit in enumerate(anc_X_qubits):
                    lookback = -num_anc_X_qubits + j
                    coords = get_qubit_coords(anc_qubit)
                    coords += (0,)
                    target = [
                        stim.target_rec(lookback),
                        stim.target_rec(lookback - num_anc_qubits),
                    ]
                    synd_extr_circuit.append("DETECTOR", target, coords)

            if p_reset > 0:
                synd_extr_circuit.append("X_ERROR", anc_Z_qids, p_reset)
                synd_extr_circuit.append("Z_ERROR", anc_X_qids, p_reset)
            if p_idle > 0:
                synd_extr_circuit.append("DEPOLARIZE1", data_qids, p_idle)
            if p_bitflip > 0:
                synd_extr_circuit.append("X_ERROR", data_qids, p_bitflip)

            # if custom_noise_channel is not None:
            #     synd_extr_circuit.append(custom_noise_channel[0],
            #                              data_qids,
            #                              custom_noise_channel[1])

            synd_extr_circuit.append("TICK")
            synd_extr_circuit.append("SHIFT_COORDS", (), (0, 0, 1))

            return synd_extr_circuit

        # Main circuit
        for qubit in tanner_graph.vs:
            coords = get_qubit_coords(qubit)
            circuit.append("QUBIT_COORDS", qubit.index, coords)

        # Initialize qubits
        if temp_bdrys == "z":
            circuit.append("RZ", data_qids)
            if p_reset > 0 and not perfect_init_final:
                circuit.append("X_ERROR", data_qids, p_reset)

        elif temp_bdrys == "r":
            circuit.append("RX", data_q1s)
            circuit.append("RZ", data_q2s)
            if p_reset > 0 and not perfect_init_final:
                circuit.append("Z_ERROR", data_q1s, p_reset)
                circuit.append("X_ERROR", data_q2s, p_reset)

            circuit.append("TICK")

            circuit.append("CX", red_links.ravel())
            if p_cnot > 0:
                circuit.append("DEPOLARIZE2", red_links.ravel(), p_cnot)

        else:
            # Data qubits inside the initial patch
            data_qids_init_patch = data_qubits.select(y_ge=y_init_patch_bdry)["qid"]
            circuit.append("RZ", data_qids_init_patch)
            if p_reset > 0 and not perfect_init_final:
                circuit.append("X_ERROR", data_qids_init_patch, p_reset)

            # Data qubits outside the initial patch
            circuit.append("RX", data_q1s)
            circuit.append("RZ", data_q2s)
            if p_reset > 0:
                circuit.append("Z_ERROR", data_q1s, p_reset)
                circuit.append("X_ERROR", data_q2s, p_reset)

            circuit.append("TICK")

            circuit.append("CX", red_links.ravel())
            if p_cnot > 0:
                circuit.append("DEPOLARIZE2", red_links.ravel(), p_cnot)

        circuit.append("RZ", anc_Z_qids)
        circuit.append("RX", anc_X_qids)

        if p_reset > 0:
            circuit.append("X_ERROR", anc_Z_qids, p_reset)
            circuit.append("Z_ERROR", anc_X_qids, p_reset)

        if p_bitflip > 0:
            circuit.append("X_ERROR", data_qids, p_bitflip)

        # if custom_noise_channel is not None:
        #     circuit.append(custom_noise_channel[0],
        #                    data_qids,
        #                    custom_noise_channel[1])

        circuit.append("TICK")

        circuit += get_synd_extr_circuit(first=True)
        circuit += get_synd_extr_circuit() * (rounds - 1)

        # Final data qubit measurements (& observables for red boundaries)
        p_meas_final = 0 if perfect_init_final else p_meas
        if temp_bdrys in {"z", "mixed"}:
            circuit.append("MZ", data_qids, p_meas_final)
            if use_last_detectors:
                for j_anc, anc_qubit in enumerate(anc_Z_qubits):
                    anc_qubit: ig.Vertex
                    ngh_data_qubits = anc_qubit.neighbors()
                    lookback_inds = [
                        -num_data_qubits + data_qids.index(q.index)
                        for q in ngh_data_qubits
                    ]
                    lookback_inds.append(-num_data_qubits - num_anc_qubits + j_anc)
                    target = [stim.target_rec(ind) for ind in lookback_inds]
                    circuit.append(
                        "DETECTOR", target, get_qubit_coords(anc_qubit) + (0,)
                    )

        else:
            if not use_last_detectors:
                raise NotImplementedError

            circuit.append("CX", red_links.ravel())
            if p_cnot > 0 and not perfect_init_final:
                circuit.append("DEPOLARIZE2", red_links.ravel(), p_cnot)

            circuit.append("TICK")

            circuit.append("MZ", data_q2s, p_meas_final)  # ZZ measurement outcomes

            num_data_q2s = data_q2s.size
            lookback_inds_anc = {}
            for j, data_q2 in enumerate(data_q2s):
                for anc_Z_qubit in tanner_graph.vs[data_q2].neighbors():
                    if anc_Z_qubit["pauli"] == "Z" and anc_Z_qubit["color"] != "r":
                        anc_Z_qid = anc_Z_qubit.index
                        lookback_ind = j - num_data_q2s
                        try:
                            lookback_inds_anc[anc_Z_qid].append(lookback_ind)
                        except KeyError:
                            lookback_inds_anc[anc_Z_qid] = [lookback_ind]

            obs_Z_lookback_inds = []
            for j_anc_Z, anc_Z_qubit in enumerate(anc_Z_qubits):
                check_meas_lookback_ind = j_anc_Z - num_data_q2s - num_anc_qubits
                if anc_Z_qubit["color"] != "g":
                    obs_Z_lookback_inds.append(check_meas_lookback_ind)
                try:
                    lookback_inds = lookback_inds_anc[anc_Z_qubit.index]
                except KeyError:
                    continue
                lookback_inds.append(check_meas_lookback_ind)
                target = [stim.target_rec(ind) for ind in lookback_inds]
                circuit.append("DETECTOR", target, get_qubit_coords(anc_Z_qubit) + (0,))

            target = [stim.target_rec(ind) for ind in obs_Z_lookback_inds]
            if self.logical_gap:
                raise NotImplementedError
            else:
                circuit.append("OBSERVABLE_INCLUDE", target, 0)

            circuit.append("MX", data_q1s, p_meas_final)  # XX measurement outcomes

            num_data_q1s = data_q1s.size
            lookback_inds_anc = {}
            for j, data_q1 in enumerate(data_q1s):
                for anc_X_qubit in tanner_graph.vs[data_q1].neighbors():
                    if anc_X_qubit["pauli"] == "X" and anc_X_qubit["color"] != "r":
                        anc_X_qid = anc_X_qubit.index
                        lookback_ind = j - num_data_q1s
                        try:
                            lookback_inds_anc[anc_X_qid].append(lookback_ind)
                        except KeyError:
                            lookback_inds_anc[anc_X_qid] = [lookback_ind]

            obs_X_lookback_inds = []
            for j_anc_X, anc_X_qubit in enumerate(anc_X_qubits):
                check_meas_lookback_ind = (
                    j_anc_X - num_data_q1s - num_data_q2s - num_anc_X_qubits
                )
                if anc_X_qubit["color"] != "g":
                    obs_X_lookback_inds.append(check_meas_lookback_ind)

                try:
                    lookback_inds = lookback_inds_anc[anc_X_qubit.index]
                except KeyError:
                    continue

                lookback_inds.append(check_meas_lookback_ind)
                target = [stim.target_rec(ind) for ind in lookback_inds]
                circuit.append("DETECTOR", target, get_qubit_coords(anc_X_qubit) + (0,))

            target = [stim.target_rec(ind) for ind in obs_X_lookback_inds]
            if self.logical_gap:
                raise NotImplementedError
            else:
                circuit.append("OBSERVABLE_INCLUDE", target, 1)

        # Logical observables
        if temp_bdrys in {"z", "mixed"}:
            if self.shape in {"tri", "growing"}:
                qubits_logs = [tanner_graph.vs.select(obs=True)]

            elif self.shape == "rec":
                qubits_log_r = tanner_graph.vs.select(obs_r=True)
                qubits_log_g = tanner_graph.vs.select(obs_g=True)
                qubits_logs = [qubits_log_r, qubits_log_g]

            for obs_id, qubits_log in enumerate(qubits_logs):
                lookback_inds = [
                    -num_data_qubits + data_qids.index(q.index) for q in qubits_log
                ]
                target = [stim.target_rec(ind) for ind in lookback_inds]
                circuit.append("OBSERVABLE_INCLUDE", target, obs_id)
                if self.logical_gap:
                    circuit.append("DETECTOR", target, obs_id)

    def get_detector(self, detector_id):
        """
        Get the ancillary qubit and round corresponding to a detector from a
        given detector ID.

        Parameters
        ----------
        detector_id : int
            Detector ID.

        Returns
        -------
        anc : ig.Vertex
            Ancillary qubit involved in the detector.
        round : int
            Round that the detector belongs to.
        """
        return self.detectors[detector_id]
    
#############################################################################################

def get_color_code(d=5,rounds=5):
    colorcode = ColorCode(
    d=d,
    rounds=rounds,
    cnot_schedule="tri_optimal",  # Default CNOT schedule optimized in our paper.
    p_circuit=0,
    )
    return StimCodeCircuit(stim_circuit = colorcode.circuit)



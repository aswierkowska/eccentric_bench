import sys
import os
import stim

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

import yaml
import logging

from itertools import product
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from qiskit.compiler import transpile
from qiskit_qec.utils import get_stim_circuits
from qiskit_qec.circuits import StimCodeCircuit
from backends import get_backend, QubitTracking
from codes import get_code, get_max_d, get_min_n
from noise import get_noise_model
from decoders import decode
from transpilers import run_transpiler, translate
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging
from metrics.utils import *
import stim

from tqec import compile_block_graph, NoiseModel
from tqec.utils.enums import Basis
from tqec.computation.block_graph import BlockGraph, BlockKind, block_kind_from_str
from tqec.computation.cube import CubeKind, Port, YHalfCube
from tqec.computation.pipe import PipeKind
from tqec.computation.cube import ZXCube
from tqec.utils.position import FloatPosition3D, Position3D
from tqec.utils.scale import round_or_fail
from tqec.gallery import cnot, three_cnots, memory, stability, cz
from tqec.gallery.steane_encoding import steane_encoding
from tqec.utils.position import Direction3D, Position3D, SignedDirection3D

def single_cnot_full_memory(self, distance_scale: int = 1, n1: int = 1, cycles: int = 1):
    g = BlockGraph("Logical CNOT")
    cnot_counter = 0
    for _ in range(n1):
        nodes = [
            (Position3D(placement_x, placement_y, 0), "P", f"In_Control_{cnot_counter}"),
            (Position3D(placement_x, placement_y, 1), "ZXX", ""),
            (Position3D(placement_x, placement_y, 2), "ZXZ", ""),
            (Position3D(placement_x, placement_y, 3), "P", f"Out_Control_{cnot_counter}"),
            (Position3D(placement_x, placement_y+1, 1), "ZXX", ""),
            (Position3D(placement_x, placement_y+1, 2), "ZXZ", ""),
            (Position3D(placement_x+1, placement_y+1, 0), "P", f"In_Target_{cnot_counter}"),
            (Position3D(placement_x+1, placement_y+1, 1), "ZXZ", ""),
            (Position3D(placement_x+1, placement_y+1, 2), "ZXZ", ""),
            (Position3D(placement_x+1, placement_y+1, 3), "P", f"Out_Target_{cnot_counter}"),
        ]
        for pos, kind, label in nodes:
            g.add_cube(pos, kind, label)
            pipes = [(0, 1), (1, 2), (2, 3), 
                (1, 4), (4, 5), (5, 8),
                (6, 7), (7, 8), (8, 9)
        ]
        for p0, p1 in pipes:
            g.add_pipe(nodes[p0][0], nodes[p1][0])
        g.fill_ports(ZXCube.from_str("ZXZ"))
        placement_x += 2
    compiled_graph = compile_block_graph(g)
    stim_circuit = compiled_graph.generate_stim_circuit(
        k = distance_scale, manhattan_radius=2
    )

    return stim_circuit

def single_cnot_n_rounds(
        self,
        distance_scale: int = 1,
        num_memory_rounds: int = 3,
):
    """
    Builds:
        - One CNOT interaction layer
        - Followed by N syndrome measurement memory rounds
        - On both control and target
    """

    g = BlockGraph("Logical CNOT + N memory rounds")

    cnot_counter = 0
    nodes = []

    # =========================
    # CONTROL PATCH
    # =========================
    control_nodes = []

    # Input
    control_nodes.append(
        (Position3D(0, 0, 0), "P", f"In_Control_{cnot_counter}")
    )

    # CNOT interaction layer
    control_nodes.append(
        (Position3D(0, 0, 1), "ZXX", "")
    )

    # Memory rounds after CNOT
    for r in range(num_memory_rounds):
        z = 2 + r
        control_nodes.append(
            (Position3D(0, 0, z), "ZXZ", "")
        )

    # Output
    control_out_z = 2 + num_memory_rounds
    control_nodes.append(
        (Position3D(0, 0, control_out_z), "P",
         f"Out_Control_{cnot_counter}")
    )

    nodes.extend(control_nodes)

    # =========================
    # TARGET PATCH
    # =========================
    target_nodes = []

    target_nodes.append(
        (Position3D(1, 1, 0), "P", f"In_Target_{cnot_counter}")
    )

    # CNOT interaction layer
    target_nodes.append(
        (Position3D(1, 1, 1), "ZXZ", "")
    )

    # Memory rounds
    for r in range(num_memory_rounds):
        z = 2 + r
        target_nodes.append(
            (Position3D(1, 1, z), "ZXZ", "")
        )

    target_out_z = 2 + num_memory_rounds
    target_nodes.append(
        (Position3D(1, 1, target_out_z), "P",
         f"Out_Target_{cnot_counter}")
    )

    nodes.extend(target_nodes)

    # =========================
    # ANCILLA COLUMN (fixed)
    # =========================
    ancilla_nodes = [
        (Position3D(0, 1, 1), "ZXX", ""),
        (Position3D(0, 1, 2), "ZXZ", ""),
    ]
    nodes.extend(ancilla_nodes)

    # =========================
    # ADD CUBES
    # =========================
    for pos, kind, label in nodes:
        g.add_cube(pos, kind, label)

    # =========================
    # ADD VERTICAL PIPES
    # =========================
    def connect_column(column_nodes):
        for i in range(len(column_nodes) - 1):
            g.add_pipe(column_nodes[i][0], column_nodes[i + 1][0])

    connect_column(control_nodes)
    connect_column(target_nodes)

    # =========================
    # CNOT INTERACTION PIPES
    # =========================
    # Control z=1 -> Ancilla z=1
    g.add_pipe(control_nodes[1][0], ancilla_nodes[0][0])

    # Ancilla z=1 -> z=2
    g.add_pipe(ancilla_nodes[0][0], ancilla_nodes[1][0])

    # =========================
    # Finalize
    # =========================
    g.fill_ports(ZXCube.from_str("ZXZ"))

    compiled_graph = compile_block_graph(g)
    stim_circuit = compiled_graph.generate_stim_circuit(
        k=distance_scale,
        manhattan_radius=2
    )

    return stim_circuit


def run_experiment(
    experiment_name,
    backend_name,
    backend_size,
    code_name,
    decoder,
    d,
    cycles,
    num_samples,
    error_types,
    error_prob,
    lock,
    layout_method=None,
    routing_method=None,
    translating_method=None,
):
    try:
        backend = get_backend(backend_name, backend_size)
        #if d == None:
        #    d = get_max_d(code_name, backend.coupling_map.size())
        #    print(f"Max distance for {code_name} on backend {backend_name} is {d}")
        #    if d < 3:
        #        logging.info(
        #            f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: Execution not possible"
        #        )
        #        return
        # 
        #if cycles is not None and cycles <= 1:
        #    logging.info(
        #        f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: Execution not possible, cycles must be greater than 1"
        #    )
        #    return
        
        #if cycles is None:
        #    cycles = d
        
        x = int((d - 1) / 2)
        stim_circuit = single_cnot_n_rounds(x, 1, cycles)
        code = StimCodeCircuit(stim_circuit = stim_circuit)
        detectors, logicals = code.stim_detectors()

        if translating_method:
            code.qc = translate(code.qc, translating_method)
       
        mappings = {}

        for _ in range(10):
            t = run_transpiler(code.qc, backend, layout_method, routing_method)
            if "swap" in detailed_gate_count_qiskit(t):
                mappings[detailed_gate_count_qiskit(t)["swap"]] = t
            else:
                mapping[0] = t

        code.qc = mappings[min(mappings)]
        #code.qc = run_transpiler(code.qc, backend, layout_method, routing_method)
        qt = QubitTracking(backend, code.qc)
        #stim_circuit = get_stim_circuits(
        #    code.qc, detectors=detectors, logicals=logicals
        #)[0][0]
        for error_type in error_types:
            noise_model = get_noise_model(error_type, qt, error_prob, backend)
            noisy_stim_circuit = noise_model.noisy_circuit(stim_circuit)
            
            lers = []
            for i in range(10):
                lers.append(decode(code_name, noisy_stim_circuit, num_samples, decoder, backend_name, error_type))
            logical_error_rate = min(lers)

            result_data = {
                "backend": backend_name,
                "backend_size": backend_size,
                "code": code_name,
                "decoder": decoder,
                "distance": d,
                "cycles": cycles if cycles else d,
                "num_samples": num_samples,
                "error_type": error_type,
                "error_probability": error_prob,
                "logical_error_rate": f"{logical_error_rate:.6f}",
                "layout_method": layout_method if layout_method else "N/A",
                "routing_method": routing_method if routing_method else "N/A",
                "translating_method": translating_method if translating_method else "N/A"
            }

            with lock:
                save_results_to_csv(result_data, experiment_name)


            if backend_size:
                logging.info(
                    f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name} {backend_size}, error type {error_type}, decoder {decoder}: {logical_error_rate:.6f}"
                )
            else:
                logging.info(
                    f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}, error type {error_type}, decoder {decoder}: {logical_error_rate:.6f}"
                )

    except Exception as e:
            logging.error(
                f"{experiment_name} | Failed to run experiment for {code_name}, distance {d}, backend {backend_name}, error type {error_type}: {e}"
            )


if __name__ == "__main__":
    with open("experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        num_samples = experiment["num_samples"]
        backends = experiment["backends"]
        codes = experiment["codes"]
        decoders = experiment["decoders"]
        error_types = experiment["error_types"]
        error_probabilities = experiment.get("error_probabilities", [None])
        cycles = experiment.get("cycles", [None])
        layout_methods = experiment.get("layout_methods", [None])
        routing_methods = experiment.get("routing_methods", [None])
        translating_methods = experiment.get("translating_methods", [None])

        setup_experiment_logging(experiment_name)
        save_experiment_metadata(experiment, experiment_name)
        manager = Manager()
        lock = manager.Lock()

        with ProcessPoolExecutor() as executor:
            if "backends_sizes" in experiment and "distances" in experiment:
                raise ValueError("Cannot set both backends_sizes and distances in the same experiment")
            if "distances" in experiment:
                distances = experiment["distances"]
                parameter_combinations = product(backends, codes, cycles, decoders, error_probabilities, distances, layout_methods, routing_methods, translating_methods)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        get_min_n(code_name, d),
                        code_name,
                        decoder,
                        d,
                        num_rounds,
                        num_samples,
                        error_types,
                        error_prob,
                        lock,
                        layout_method,
                        routing_method,
                        translating_method
                    )
                    for backend, code_name, num_rounds, decoder, error_prob, d, layout_method, routing_method, translating_method in parameter_combinations
                ]
            elif "backends_sizes" in experiment:
                backends_sizes = experiment["backends_sizes"]
                parameter_combinations = product(
                    backends, backends_sizes, codes, cycles, decoders, error_probabilities, layout_methods, routing_methods, translating_methods
                )
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        backends_sizes,
                        code_name,
                        decoder,
                        None,
                        num_rounds,
                        num_samples,
                        error_types,
                        error_prob,
                        lock,
                        layout_method,
                        routing_method,
                        translating_method,
                    )
                    for backend, backends_sizes, code_name, num_rounds, decoder, error_prob, layout_method, routing_method, translating_method in parameter_combinations
                ]
            else:
                parameter_combinations = product(backends, codes, cycles, decoders, error_probabilities, layout_methods, routing_methods, translating_methods)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        None,
                        code_name,
                        decoder,
                        None,
                        num_rounds,
                        num_samples,
                        error_types,
                        error_prob,
                        lock,
                        layout_method,
                        routing_method,
                        translating_method,
                    )
                    for backend, code_name, num_rounds, decoder, error_prob, layout_method, routing_method, translating_method in parameter_combinations
                ]
            for future in futures:
                future.result()

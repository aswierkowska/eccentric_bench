import sys
import os

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

import stim
import numpy as np
import yaml
import logging

from itertools import product
from concurrent.futures import ProcessPoolExecutor
from qiskit.compiler import transpile
from qiskit_qec.utils import get_stim_circuits
from backends import get_backend
from codes import get_code, get_max_d
from noise import add_stim_noise
from decoders import decode



def simulate_circuit(circuit: stim.Circuit, num_shots: int, decoder: str) -> int:
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )
    detector_error_model = circuit.detector_error_model()
    # TODO do we need those:    decompose_errors=True, approximate_disjoint_errors=True
    predictions = decode(decoder, detector_error_model, detection_events)
    # TODO: create metrics and move following there:
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors / num_shots


def run_experiment(
    experiment_name,
    backend_name,
    backend_size,
    code_name,
    decoder,
    d,
    num_samples,
    error_prob,
):
    #try:
    stim_gates = [
        "x",
        "y",
        "z",
        "cx",
        "cz",
        "cy",
        "h",
        "s",
        "s_dag",
        "swap",
        "reset",
        "measure",
        "barrier",
    ]
    backend = get_backend(backend_name, backend_size)
    if d == None:
        d = get_max_d(code_name, backend.coupling_map.size())
        if d < 3:
            logging.info(
                f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: Execution not possible"
            )
            return
    code = get_code(code_name, d)
    detectors, logicals = code.stim_detectors()
    # FOR GROSS CODE:
    # detectors, logicals = code.detectors, code.logicals
    # circuit = code.qc

    for state, qc in code.circuit.items():
        code.circuit[state] = transpile(
            code.circuit[state],
            backend=backend,
            basis_gates=stim_gates,
            optimization_level=0,
        )
        stim_circuit = get_stim_circuits(
            code.circuit[state], detectors=detectors, logicals=logicals
        )[0][0]
        stim_circuit = add_stim_noise(stim_circuit, error_prob)
        logical_error_rate = simulate_circuit(stim_circuit, num_samples, decoder)
        if backend_size:
            logging.info(
                f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name} {backend_size}, decoder {decoder}: {logical_error_rate:.6f}"
            )
        else:
            logging.info(
                f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}, decoder {decoder}: {logical_error_rate:.6f}"
            )
    #except Exception as e:
    #    logging.error(
    #        f"{experiment_name} | Failed to run experiment for {code_name}, distance {d}, backend {backend_name}: {e}"
    #    )


if __name__ == "__main__":
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.basicConfig(
        filename="qecc_benchmark.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    with open("experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        num_samples = experiment["num_samples"]
        backends = experiment["backends"]
        codes = experiment["codes"]
        decoders = experiment["decoders"]
        error_prob = experiment["error_probability"]

        # TODO: better handling case if distances and backends_sizes are both set

        with ProcessPoolExecutor() as executor:
            if "distances" in experiment:
                distances = experiment["distances"]
                parameter_combinations = product(backends, codes, decoders, distances)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        None,
                        code_name,
                        decoder,
                        d,
                        num_samples,
                        error_prob,
                    )
                    for backend, code_name, decoder, d in parameter_combinations
                ]
            elif "backends_sizes" in experiment:
                backends_sizes = experiment["backends_sizes"]
                parameter_combinations = product(
                    backends, backends_sizes, codes, decoders
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
                        num_samples,
                        error_prob,
                    )
                    for backend, backends_sizes, code_name, decoder in parameter_combinations
                ]
            else:
                parameter_combinations = product(backends, codes, decoders)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        None,
                        code_name,
                        decoder,
                        None,
                        num_samples,
                        error_prob,
                    )
                    for backend, code_name, decoder in parameter_combinations
                ]

            for future in futures:
                future.result()
import sys
import os
import stim

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

import yaml
import logging

from itertools import product
from concurrent.futures import ProcessPoolExecutor
from qiskit.compiler import transpile
from qiskit_qec.utils import get_stim_circuits
from backends import get_backend, QubitTracking
from codes import get_code, get_max_d, get_min_n, make_color_code_circuit
from noise import get_noise_model
from decoders import decode
from transpilers import run_transpiler, translate
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging
import stim
from qiskit_qec.circuits import StimCodeCircuit

def run_experiment(
    experiment_name,
    backend_name,
    backend_size,
    code_name,
    decoder,
    d,
    cycles,
    num_samples,
    error_type,
    error_prob,
    layout_method=None,
    routing_method=None,
    translating_method=None,
):
    try:
        stim_circuit = make_color_code_circuit(
            obs_basis='Z',
            base_data_width=d,
            rounds=d,
            noise_strength=error_prob,
        )
        noise_model = get_noise_model(error_type, None, error_prob, None)
        stim_circuit = noise_model.noisy_circuit(stim_circuit)


        logical_error_rate = decode(code_name, stim_circuit, num_samples, decoder)

        if logical_error_rate == None:
            exit(1)

        result_data = {
            "code": code_name,
            "distance": d,
            "cycles": cycles if cycles else d,
            "decoder": decoder,
            "num_samples": num_samples,
            "error_type": error_type,
            "error_probability": error_prob,
            "logical_error_rate": f"{logical_error_rate:.6f}",
            "layout_method": layout_method if layout_method else "N/A",
            "routing_method": routing_method if routing_method else "N/A",
            "translating_method": translating_method if translating_method else "N/A"
        }

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
    with open("experiments_chromobius.yaml", "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        num_samples = experiment["num_samples"]
        backends = experiment["backends"]
        codes = experiment["codes"]
        decoders = experiment["decoders"]
        error_types = experiment["error_types"]
        error_probabilities = experiment.get("error_probabilities", [None])
        cycles = experiment.get("cycles", None)
        layout_methods = experiment.get("layout_methods", [None])
        routing_methods = experiment.get("routing_methods", [None])
        translating_methods = experiment.get("translating_methods", [None])

        setup_experiment_logging(experiment_name)
        save_experiment_metadata(experiment, experiment_name)
        # TODO: better handling case if distances and backends_sizes are both set

        with ProcessPoolExecutor() as executor:
            #if "backends_sizes" in experiment and "distances" in experiment:
            #    raise ValueError("Cannot set both backends_sizes and distances in the same experiment")
            if "distances" in experiment:
                distances = experiment["distances"]
                parameter_combinations = product(backends, codes, decoders, error_types, error_probabilities, distances, layout_methods, routing_methods, translating_methods)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        get_min_n(code_name, d),
                        code_name,
                        decoder,
                        d,
                        cycles,
                        num_samples,
                        error_type,
                        error_prob,
                        layout_method,
                        routing_method,
                        translating_method
                    )
                    for backend, code_name, decoder, error_type, error_prob, d, layout_method, routing_method, translating_method in parameter_combinations
                ]
            elif "backends_sizes" in experiment:
                backends_sizes = experiment["backends_sizes"]
                parameter_combinations = product(
                    backends, backends_sizes, codes, decoders, error_types, error_probabilities, layout_methods, routing_methods, translating_methods
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
                        cycles,
                        num_samples,
                        error_type,
                        error_prob,
                        layout_method,
                        routing_method,
                        translating_method
                    )
                    for backend, backends_sizes, code_name, decoder, error_type, error_prob, layout_method, routing_method, translating_method in parameter_combinations
                ]
            else:
                parameter_combinations = product(backends, codes, decoders, error_types, error_probabilities, layout_methods, routing_methods, translating_methods)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        None,
                        code_name,
                        decoder,
                        None,
                        cycles,
                        num_samples,
                        error_type,
                        error_prob,
                        layout_method,
                        routing_method,
                        translating_method
                    )
                    for backend, code_name, decoder, error_type, error_prob, layout_method, routing_method, translating_method in parameter_combinations
                ]
            for future in futures:
                future.result()

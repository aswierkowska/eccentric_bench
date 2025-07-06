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
from backends import get_backend, QubitTracking
from codes import get_code, get_max_d, get_min_n, make_idle_qubit_circuit
from noise import get_noise_model
from decoders import decode, raw_error_rate
from transpilers import run_transpiler, translate
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging
import stim

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
    lock,
    layout_method=None,
    routing_method=None,
    translating_method=None,
):
    try:
        print("Starting experiment")
        backend = get_backend(backend_name, backend_size)
        print("Got backend")
        if d == None:
            d = get_max_d(code_name, backend.coupling_map.size())
            print(f"Max distance for {code_name} on backend {backend_name} is {d}")
            if d < 3:
                logging.info(
                    f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: Execution not possible"
                )
                return
        
        if cycles is not None and cycles <= 1:
            logging.info(
                f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: Execution not possible, cycles must be greater than 1"
            )
            return
        
        if cycles is None:
            cycles = d
        
        code = get_code(code_name, d, cycles)
        detectors, logicals = code.stim_detectors()
        idle_circuit = make_idle_qubit_circuit(cycles)
        detectors_idle, logicals_idle = idle_circuit.stim_detectors()

        code.qc = run_transpiler(code.qc, backend, layout_method, routing_method)
        idle_circuit.qc = run_transpiler(idle_circuit.qc, backend, layout_method, routing_method)

        qt = QubitTracking(backend, code.qc)
        qt_idle = QubitTracking(backend, idle_circuit.qc)

        stim_circuit = get_stim_circuits(
            code.qc, detectors=detectors, logicals=logicals
        )[0][0]
        stim_circuit_idle = get_stim_circuits(
            idle_circuit.qc, detectors=detectors_idle, logicals=logicals_idle
        )[0][0]
        print(cycles)
        print(stim_circuit_idle)

        noise_model = get_noise_model(error_type, qt, error_prob, backend)
        noise_model_idle = get_noise_model(error_type, qt_idle, error_prob, backend)

        stim_circuit = noise_model.noisy_circuit(stim_circuit)
        stim_circuit_idle = noise_model_idle.noisy_circuit(stim_circuit_idle)
        
        raw_error = raw_error_rate(stim_circuit, num_samples) / num_samples
        idle_error = raw_error_rate(stim_circuit_idle, num_samples) / num_samples
        corrected_error = decode(code_name, stim_circuit, num_samples, decoder, backend_name, error_type) / num_samples


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
            "raw_error_rate": f"{raw_error:.6f}",
            "idle_error_rate": f"{idle_error:.6f}",
            "corrected_error_rate": f"{corrected_error:.6f}",
        }

        with lock:
            save_results_to_csv(result_data, experiment_name)


        if backend_size:
            logging.info(
                f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name} {backend_size}, error type {error_type}, decoder {decoder}: raw: {raw_error:.6f} idle_error_rate: {idle_error:.6f} corrected_error_rate: {corrected_error:.6f}"
            )
        else:
            logging.info(
                f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}, error type {error_type}, decoder {decoder}: raw: {raw_error:.6f} idle_error_rate: {idle_error:.6f} corrected_error_rate: {corrected_error:.6f}"
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
        cycles = experiment.get("cycles", None)
        layout_methods = experiment.get("layout_methods", [None])
        routing_methods = experiment.get("routing_methods", [None])
        translating_methods = experiment.get("translating_methods", [None])

        setup_experiment_logging(experiment_name)
        save_experiment_metadata(experiment, experiment_name)
        manager = Manager()
        lock = manager.Lock()
        # TODO: better handling case if distances and backends_sizes are both set

        with ProcessPoolExecutor() as executor:
            if "backends_sizes" in experiment and "distances" in experiment:
                raise ValueError("Cannot set both backends_sizes and distances in the same experiment")
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
                        lock,
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
                        lock,
                        layout_method,
                        routing_method,
                        translating_method,
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
                        lock,
                        layout_method,
                        routing_method,
                        translating_method,
                    )
                    for backend, code_name, decoder, error_type, error_prob, layout_method, routing_method, translating_method in parameter_combinations
                ]
            for future in futures:
                future.result()

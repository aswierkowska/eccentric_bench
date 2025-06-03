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
from codes import get_code, get_max_d
from noise import get_noise_model
from decoders import decode
from transpilers import run_transpiler, translate
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging

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
        print("Starting experiment")
        backend = get_backend(backend_name, backend_size)
        print("Got backend")
        if d == None:
            d = get_max_d(code_name, backend.coupling_map.size())
            if d < 3:
                logging.info(
                    f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: Execution not possible"
                )
                return
        code = get_code(code_name, d, cycles)
        detectors, logicals = code.stim_detectors()

        for state, qc in code.circuit.items():
            print("Before translating")
            #tmp_stim_circuit = get_stim_circuits(
            #    code.circuit[state], detectors=detectors, logicals=logicals
            #)[0][0]
            #tmp_stim_circuit.to_file(f'our_gross_qiskit_{state}.stim')
            if translating_method:
                code.circuit[state] = translate(code.circuit[state], translating_method)
            # TODO: either else here or sth
            print("Before transpiler")
            #code.circuit[state] = run_transpiler(code.circuit[state], backend, layout_method, routing_method)
            print("After transpiler")
            #qt = QubitTracking(backend, code.circuit[state])
            print("After QT")
            #stim_circuit = get_stim_circuits(
            #    code.circuit[state], detectors=detectors, logicals=logicals
            #)[0][0]
            #stim_circuit.to_file(f'our_gross_transpiled_{state}.stim')
            print("After GET STIM CIRCUIT")
            #noise_model = get_noise_model(error_type, qt, error_prob, backend)
            print("After get_noise_model")
            #stim_circuit = noise_model.noisy_circuit(stim_circuit)
            #stim_circuit.to_file(f'our_gross_noist_{state}.stim')
            #fname = code_name + ".stim"
            #stim_circuit.to_file(fname)
            print("After adding noise")

            # TODO let's put the og code with 0.004 error here
            stim_circuit = stim.Circuit.from_file("gdg_original_gross.stim")
            print("before decoding")
            logical_error_rate = decode(code_name, stim_circuit, num_samples, decoder)
            print("After decoding")

            if logical_error_rate == None:
                exit(1)

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
        cycles = experiment.get("cycles", None)
        layout_methods = experiment.get("layout_methods", [None])
        routing_methods = experiment.get("routing_methods", [None])
        translating_methods = experiment.get("translating_methods", [None])

        setup_experiment_logging(experiment_name)
        save_experiment_metadata(experiment, experiment_name)
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
                        None,
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

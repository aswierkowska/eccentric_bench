import sys
import os

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

import yaml
import logging

from itertools import product
from concurrent.futures import ProcessPoolExecutor
from qiskit.compiler import transpile
from qiskit_qec.utils import get_stim_circuits
from backends import get_backend
from codes import get_code, get_max_d
from noise import get_noise_model
from decoders import decode


def run_experiment(
    experiment_name,
    backend_name,
    backend_size,
    code_name,
    decoder,
    d,
    cycles,
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
      #  "swap",
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
    code = get_code(code_name, d, cycles)
    detectors, logicals = code.stim_detectors()

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

        noise_model = get_noise_model("", error_prob)
        stim_circuit = noise_model.noisy_circuit(stim_circuit)
        #if hasattr(backend, 'add_realistic_noise'): 
        #    stim_circuit = backend.add_realistic_noise(stim_circuit)
        #else:
        #    stim_circuit = add_stim_noise(stim_circuit, error_prob, error_prob, error_prob, error_prob)
        logical_error_rate = decode(code_name, stim_circuit, num_samples, decoder)
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

        if "cycles" in experiment:
            cycles = experiment["cycles"]
        else:
            cycles = None

        print(experiment)

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
                        cycles,
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
                        cycles,
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
                        cycles,
                        num_samples,
                        error_prob,
                    )
                    for backend, code_name, decoder in parameter_combinations
                ]

            for future in futures:
                future.result()
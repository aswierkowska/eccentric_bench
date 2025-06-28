import sys
import os
import yaml
import logging
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

import stim

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

from backends import get_backend
from codes import get_code, get_max_d
from transpilers import run_transpiler
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging
from metrics import count_total_gates_qiskit, get_resource_overhead_total_gates, get_resource_overhead_2q_gates, count_swaps_qiskit


def run_experiment(
    experiment_name,
    num_samples,
    backend_name,
    backend_size,
    code_name,
    d,  # d is None so it'll be calculated dynamically
    cycles,
    lock,
    routing_method,
    layout_method
):
    try:
        print(f'{code_name} {routing_method} {backend_name} starts.')
        backend = get_backend(backend_name, backend_size)
        if d is None:
            d = get_max_d(code_name, backend.coupling_map.size())
            if d < 3:
                logging.info(
                    f"{experiment_name} | Skipping {code_name} with distance {d} on backend {backend_name}: distance too small"
                )
                return

        code = get_code(code_name, d, cycles)

        for state, qc in code.circuit.items():
            original_circuit = qc
            #original_total_gates = count_total_gates_qiskit(original_circuit)

            gates = []
            tq_gates = []
            swaps = []

            for _ in range(num_samples):
                transpiled_circuit = run_transpiler(original_circuit, backend, layout_method, routing_method)
                gates.append(get_resource_overhead_total_gates(original_circuit, transpiled_circuit))
                tq_gates.append(get_resource_overhead_2q_gates(original_circuit, transpiled_circuit))
                swaps.append(count_swaps_qiskit(transpiled_circuit))
            gates = np.array(gates)
            tq_gates = np.array(tq_gates)
            swaps = np.array(swaps)
            result_data = {
                        "backend": backend_name,
                        "backend_size": backend_size,
                        "code": code_name,
                        "distance": d,
                        "cycles": cycles if cycles else d,
                        "state": state,
                        "routing_method": routing_method,
                        "layout_method": layout_method,
                        "gate_overhead_mean": np.mean(gates),
                        "gate_overhead_var": np.var(gates),
                        "tq_gate_overhead_mean": np.mean(tq_gates),
                        "tq_gate_overhead_var": np.var(tq_gates),
                        "swap_overhead_mean": np.mean(swaps),
                        "swap_overhead_var": np.var(swaps),
            }
            print(f'{code_name} is finished.')
            with lock:
                save_results_to_csv(result_data, experiment_name)

    except Exception as e:
        logging.error(
            f"{experiment_name} | Failed to run experiment for code {code_name}, backend {backend_name}: {e}"
        )


if __name__ == "__main__":
    with open("routing_experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        num_samples = experiment["num_samples"]
        backends = experiment["backends"]
        backends_sizes = experiment["backends_sizes"]
        codes = experiment["codes"]
        cycles = experiment.get("cycles", None)
        backends_sizes = experiment["backends_sizes"]
        routing_methods = experiment.get("routing_methods", [None])
        layout_methods = experiment.get("layout_methods", [None])

        setup_experiment_logging(experiment_name)
        save_experiment_metadata(experiment, experiment_name)
        manager = Manager()
        lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=1) as executor:
            parameter_combinations = product(
                backends, backends_sizes, codes, routing_methods, layout_methods
            )
            futures = [
                executor.submit(
                    run_experiment,
                    experiment_name,
                    num_samples,
                    backend,
                    backend_size,
                    code_name,
                    None,  # d is None so it'll be calculated dynamically
                    cycles,
                    lock,
                    routing_method,
                    layout_method,
                )
                for backend, backend_size, code_name, routing_method, layout_method in parameter_combinations
            ]
            for future in futures:
                future.result()

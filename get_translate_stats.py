import sys
import os
import yaml
import logging
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pytket.qasm import circuit_to_qasm

import stim
import qiskit

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

from backends import get_backend
from codes import get_code, get_max_d
from transpilers import translate
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging
from metrics import count_total_gates_qiskit, count_2q_gates_qiskit, get_resource_overhead_total_gates, get_resource_overhead_2q_gates


def run_experiment(
    experiment_name,
    num_samples,
    backend_name,
    backend_size,
    code_name,
    d,
    cycles,
    lock,
    translating_methods,
    gate_sets,
):
    try:
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
            original_total_gates = count_total_gates_qiskit(original_circuit)
            original_2q_gates = count_2q_gates_qiskit(original_circuit)

            for translating_method in translating_methods:
                for gate_set in gate_sets:
                    gates = []
                    tq_gates = []
                    for _ in range(num_samples):
                        transpiled_circuit = translate(original_circuit, translating_method, gate_set)
                        #transpiled_total_gates = count_total_gates_qiskit(transpiled_circuit)
                        gates.append(get_resource_overhead_total_gates(original_circuit, transpiled_circuit))
                        tq_gates.append(get_resource_overhead_2q_gates(original_circuit, transpiled_circuit))

                    gates = np.array(gates)
                    tq_gates = np.array(tq_gates)
                    result_data = {
                        "backend": backend_name,
                        "backend_size": backend_size,
                        "code": code_name,
                        "distance": d,
                        "cycles": cycles if cycles else d,
                        "state": state,
                        "translating_method": translating_method or "N/A",
                        "gate_set": gate_set or "N/A",
                        "original_total_gates": original_total_gates,
                        "original_2q_gates": original_2q_gates,
                        "gate_overhead_mean": np.mean(gates),
                        "gate_overhead_var": np.var(gates),
                        "tq_gate_overhead_mean": np.mean(tq_gates),
                        "tq_gate_overhead_var": np.var(tq_gates),
                    }

                    with lock:
                        save_results_to_csv(result_data, experiment_name)

    except Exception as e:
        logging.error(
            f"{experiment_name} | Failed to run experiment for code {code_name}, backend {backend_name}: {e}"
        )


if __name__ == "__main__":
    with open("translate_experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        num_samples = experiment["num_samples"]
        backends = experiment["backends"]
        codes = experiment["codes"]
        cycles = experiment.get("cycles", None)
        backends_sizes = experiment["backends_sizes"]
        translating_methods = experiment.get("translating_methods", [None])
        gate_sets = experiment.get("gate_sets", [None])

        setup_experiment_logging(experiment_name)
        save_experiment_metadata(experiment, experiment_name)
        manager = Manager()
        lock = manager.Lock()

        with ProcessPoolExecutor() as executor:
            parameter_combinations = product(
                backends, backends_sizes, codes
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
                    translating_methods,
                    gate_sets,
                )
                for backend, backend_size, code_name in parameter_combinations
            ]
            for future in futures:
                future.result()

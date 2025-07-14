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
from codes import get_code, get_max_d, get_min_n
from noise import get_noise_model
from decoders import decode
from transpilers import run_transpiler, translate
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging
from metrics import count_1q_gates_qiskit, count_2q_gates_qiskit, count_measurements_qiskit
import stim

def run_experiment():
    experiment_name = "Codes_Statistics"
    codes = ["surface"] #["surface", "hh", "gross", "color", "steane", "bacon"]
    distances = [11] #[11, 11, 12, 11, 9, 11]
    cycles = 3

    for code_name, d in zip(codes, distances):
    
        code = get_code(code_name, d, cycles)

        if code_name == "bacon":
            code_name = "bacon_Z_noCZ"

        result_data = {
            "code": code_name,
            "distance": d,
            "cycles": cycles if cycles else d,
            "num_qubits": code.qc.num_qubits,
            "1q": count_1q_gates_qiskit(code.qc),
            "2q": count_2q_gates_qiskit(code.qc),
            "measurements": count_measurements_qiskit(code.qc),
        }

        save_results_to_csv(result_data, experiment_name)


if __name__ == "__main__":
    run_experiment()
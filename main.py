import sys
import os
sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

import stim
import pymatching
import numpy as np
import yaml
import math
import logging
import matplotlib.pyplot as plt

from itertools import product
from concurrent.futures import ProcessPoolExecutor
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_qec.circuits import SurfaceCodeCircuit, CSSCodeCircuit
from qiskit_qec.codes.hhc import HHC
from qiskit_qec.utils import get_stim_circuits, noisify_circuit
from qiskit_qec.noise import PauliNoiseModel
from backends import get_custom_backend

#def get_code(code: str, d: int):
#    surface_code = CSSCode.from_code_name("surface", 3)
#    ft_prep = gate_optimal_prep_circuit(surface_code, zero_state=True, max_timeout=2)
#    return ft_prep.circ

def get_code(code_name: str, d: int):
    if code_name == "hh":
        code = HHC(d)
        css_code = CSSCodeCircuit(code, T=d)
        return css_code
    elif code_name == "surface":
        code = SurfaceCodeCircuit(d=d, T=d)
        return code
    


def get_backend(backend: str, backend_size: int):
    backed_type = backend.split("_")[0]
    if backed_type == "ibm":
        service = QiskitRuntimeService(instance="ibm-q/open/main")
        backend = service.backend(backend)
    elif backed_type == "custom" and backend_size:
        shape =  backend.split("_")[1]
        backend = get_custom_backend(shape, backend_size)
    return backend


def simulate_circuit(circuit: stim.Circuit, num_shots: int) -> int:
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    detector_error_model = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    predictions = matcher.decode_batch(detection_events)
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors / num_shots

def generate_pauli_error(p: int) -> PauliNoiseModel:
    pnm = PauliNoiseModel()
    pnm.add_operation("h", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p}) # here the weights do NOT need to be normalized
    #pnm.add_operation("cx", {"ix": 1, "xi": 1, "xx": 1})
    #pnm.add_operation("id", {"x": 1})
    #pnm.add_operation("reset", {"x": 1})
    pnm.add_operation("measure", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p})
    #pnm.add_operation("x", {"x": 1, "y": 1, "z": 1})
    return pnm

def generate_circuit_specific_pauli_error(gates: list, p: int) -> PauliNoiseModel:
    pnm = PauliNoiseModel()
    for gate in gates:
        if gate in ["cx", "swap"]:
            pnm.add_operation(gate, {"ix": p / 3, "xi": p / 3, "xx": p / 3, "ii": 1 - p})
        else:
            pnm.add_operation(gate, {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p})
    return pnm

def get_max_d(code_name: str, n: int):
    if code_name == "surface":
        # d**2 data qubits + d**2 - 1 ancilla qubits
        d =  math.floor(math.sqrt((n + 1) / 2))
        d = d - ((1 - d) % 2)
        return d
    elif code_name == "hh":
        # n = 5d^2 - 2d - 1 /2
        d = int((2 + math.sqrt(40 * n + 24)) / 10)
        d = d - ((1 - d) % 2)
        return d
    elif code_name == "gross":
        return math.floor(n / 2)
    return 0


def run_experiment(experiment_name, backend_name, backend_size, code_name, d, num_samples, error_prob):
        #   try:
        stim_gates = ['x', 'y', 'z', 'cx', 'cz', 'cy', 'h', 's', 's_dag', 'swap', 'reset', 'measure', 'barrier']
        backend = get_backend(backend_name, backend_size)
        if d == None:
           d = get_max_d(code_name, backend.coupling_map.size())
           if d < 3:
               logging.info(f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: Execution not possible")
               return
        print(code_name, d)
        code = get_code(code_name, d)
        detectors, logicals = code.stim_detectors()
        print(code.circuit['0'].num_qubits)
        code.circuit['0'] = transpile(code.circuit['0'], 
            backend=backend,
            basis_gates=stim_gates,
            optimization_level=0
        )
        
        gates = set([gate[0].name for gate in code.circuit['0'].data])
        print(gates)

        error_model = generate_circuit_specific_pauli_error(gates, error_prob)
        #error_model = generate_pauli_error(error_prob)

        for state, qc in code.circuit.items():
            code.noisy_circuit[state] = noisify_circuit(qc, error_model)
        #print(code.noisy_circuit['0'])
        stim_circuit = get_stim_circuits(
            code.noisy_circuit['0'], detectors=detectors, logicals=logicals
        )[0][0]
        
        logical_error_rate = simulate_circuit(stim_circuit, num_samples)
        if backend_size:
            logging.info(f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name} {backend_size}: {logical_error_rate}")
        else:
            logging.info(f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: {logical_error_rate}")
    
        #except Exception as e:
        #logging.error(f"{experiment_name} | Failed to run experiment for {code_name}, distance {d}, backend {backend_name}: {e}")

if __name__ == '__main__':
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.basicConfig(
        filename='qecc_benchmark.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    with open("experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        num_samples = experiment["num_samples"]
        backends = experiment["backends"]
        codes = experiment["codes"]
        error_prob = experiment["error_probability"]

        with ProcessPoolExecutor() as executor:
            if "distances" in experiment:
                distances = experiment["distances"]
                parameter_combinations = product(backends, codes, distances)
                futures = [
                    executor.submit(run_experiment, experiment_name, backend, None, code_name, d, num_samples, error_prob)
                    for backend, code_name, d in parameter_combinations
                ]
            elif "backends_sizes" in experiment:
                backends_sizes = experiment["backends_sizes"]
                parameter_combinations = product(backends, backends_sizes, codes)
                futures = [
                    executor.submit(run_experiment, experiment_name, backend, backends_sizes, code_name, None, num_samples, error_prob)
                    for backend, backends_sizes, code_name in parameter_combinations
                ]
            else:
                parameter_combinations = product(backends, codes)
                futures = [
                    executor.submit(run_experiment, experiment_name, backend, None, code_name, None, num_samples, error_prob)
                    for backend, code_name in parameter_combinations
                ]

            for future in futures:
                future.result()

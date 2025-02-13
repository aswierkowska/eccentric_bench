import sys
#sys.path.append("/home/aswierkowska/eccentric_bench/external/qiskit_qec/src")

import stim
import pymatching
import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import random

from itertools import product
from concurrent.futures import ProcessPoolExecutor
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_qec.circuits import SurfaceCodeCircuit, CSSCodeCircuit
from qiskit_qec.codes.hhc import HHC
from qiskit_qec.utils import get_stim_circuits, noisify_circuit
from qiskit_qec.noise import PauliNoiseModel
from qiskit_qec.codes.gross_code import GrossCode
from qiskit_qec.circuits.gross_code_circuit import GrossCodeCircuit

from custom_backend import FakeLargeBackend

from stimbposd import BPOSD#doesn't work with current ldpcv2 code  pip install -U ldpc==0.1.60



def load_IBM_account():
    load_dotenv()
    token=os.getenv("IBM_TOKEN")
    QiskitRuntimeService.save_account(
    token=token,
    channel="ibm_quantum" # `channel` distinguishes between different account types
    )


def get_code(code_name: str, d: int, depol_error: float = 0.00):
    if code_name == "hh":
        code = HHC(d)
        css_code = CSSCodeCircuit(code, T=d)
        return css_code
    elif code_name == "gross":
        code = GrossCode(d)
        code_circuit = GrossCodeCircuit(code, T=d, depol_error_rate=depol_error)
        return code_circuit
    elif code_name == "surface":
        code = SurfaceCodeCircuit(d=d, T=d)
    


def map_circuit(circuit: QuantumCircuit, backend: str):
    stim_gates = ['x', 'y', 'z', 'cx', 'cz', 'cy', 'h', 's', 's_dag', 'swap', 'reset', 'measure', 'barrier', 'id'] # only allows basis gates available in Stim
    if backend[:3] == "ibm":
        service = QiskitRuntimeService(instance="ibm-q/open/main")
        backend = service.backend(backend)
        circuit = transpile(circuit, 
            backend=backend,
            basis_gates=stim_gates,
            optimization_level=0
        )
    elif backend == "fake_11":
        backend = FakeLargeBackend(distance=11, number_of_chips=1)
        circuit = transpile(circuit, 
            backend=backend,
            basis_gates=stim_gates,
            optimization_level=0
        )
    return circuit


def simulate_circuit(circuit: stim.Circuit, num_shots: int) -> int:
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    np.set_printoptions(threshold=sys.maxsize)
    detector_error_model = circuit.detector_error_model(approximate_disjoint_errors=True,
                                                         allow_gauge_detectors=True,
                                                         flatten_loops=True, decompose_errors=True,
                                                         ignore_decomposition_failures=True,
                                                         block_decomposition_from_introducing_remnant_edges=True)
    matcher = BPOSD(detector_error_model, max_bp_iters=40)

    #print(detection_events[0])
    #for i in list(detection_events[0]):
    #    list(detection_events)[0][i] = False
    #print(detection_events[0])
    predictions = matcher.decode_batch(detection_events)

    random_predictions = [random.choice([[True], [False]]) for _ in range(len(observable_flips))]

    #logging.info(f"Predictions: {predictions}")

    num_mistakes = np.sum(np.any(random_predictions != observable_flips, axis=1))
    print(f"Random predictions: {num_mistakes}")
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors / num_shots

def generate_pauli_error(p: float) -> PauliNoiseModel:
    pnm = PauliNoiseModel()
    pnm.add_operation("h", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 0}) # here the weights do NOT need to be normalized
    pnm.add_operation("cx", {"ix": p/3, "xi": p/3, "xx": p/3, "ii": 0})
    pnm.add_operation("id", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 0})
    pnm.add_operation("measure", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 0})
    return pnm


def run_experiment(experiment_name, backend, code_name, d, num_samples, error_prob,depol_error=0.00):
    code = get_code(code_name, d,depol_error)
    detectors, logicals = code.stim_detectors()
    code.circuit['0'] = map_circuit(code.circuit['0'], backend)
    try:
        
        
        print("DO WE GET HERE?")
        for state, qc in code.circuit.items():
            code.noisy_circuit[state] = noisify_circuit(qc, error_prob)
        
        #circuit to pdf
        #code.noisy_circuit['0'].draw(output='mpl', filename='circuit.pdf',vertical_compression='high', scale=0.3, fold=500)
        stim_circuit = get_stim_circuits(
            code.noisy_circuit['0'], detectors=detectors, logicals=logicals
        )[0][0]
        #TODO ADD INSERT FUNCTION TO ADD DEPOL NOISE
        #stim_circuit.safe_insert("What are the paramesters")
        logical_error_rate = simulate_circuit(stim_circuit, num_samples)
        logging.info(f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend}: {logical_error_rate}")
    
    except Exception as e:
        logging.error(f"{experiment_name} | Failed to run experiment for {code_name}, distance {d}, backend {backend}: {e}")

if __name__ == '__main__':
    #load_IBM_account()

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
        distances = experiment["distances"]
        depol_error = experiment["depol_error"]
        error_prob = generate_pauli_error(experiment["error_probability"])
        print(error_prob.to_dict())

        parameter_combinations = product(backends, codes, distances)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_experiment, experiment_name, backend, code_name, d, num_samples, error_prob,depol_error)
                for backend, code_name, d in parameter_combinations
            ]

            for future in futures:
                future.result()

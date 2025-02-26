import sys
sys.path.append("/home/aswierkowska/eccentric_bench/external/qiskit_qec/src")

import stim
import pymatching
import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
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

from stimbposd import BPOSD#doesn't work with current ldpc code  pip install -U ldpc==0.1.60

#from ldpc.bplsd_decoder import BpLsdDecoder


#def get_code(code: str, d: int):
#    surface_code = CSSCode.from_code_name("surface", 3)
#    ft_prep = gate_optimal_prep_circuit(surface_code, zero_state=True, max_timeout=2)
#    return ft_prep.circ


def load_IBM_account():
    load_dotenv()
    token=os.getenv("IBM_TOKEN")
    QiskitRuntimeService.save_account(
    token=token,
    channel="ibm_quantum" # `channel` distinguishes between different account types
    )


def get_code(code_name: str, d: int):
    if code_name == "hh":
        code = HHC(d)
        css_code = CSSCodeCircuit(code, T=d)
        return css_code
    elif code_name == "gross":
        code = GrossCode(d)
        code_circuit = GrossCodeCircuit(code, T=d)
        return code_circuit
    elif code_name == "surface":
        code = SurfaceCodeCircuit(d=d, T=d)
    


def map_circuit(circuit: QuantumCircuit, backend: str):
    stim_gates = ['x', 'y', 'z', 'cx', 'cz', 'cy', 'h', 's', 's_dag', 'swap', 'reset', 'measure', 'barrier'] # only allows basis gates available in Stim
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


def generate_circuit_specific_pauli_error(gates: list, p: float) -> PauliNoiseModel:
    pnm = PauliNoiseModel()
    for gate in gates:
        if gate in ["cx", "cy", "cz", "swap"]:
            pnm.add_operation(gate, {"ix": p / 3, "xi": p / 3, "xx": p / 3, "ii": 1 - p})
        elif gate in ['x', 'y', 'z', 'h', 's', 's_dag', 'reset', 'measure']:
            pnm.add_operation(gate, {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p})
    return pnm

def simulate_circuit(circuit: stim.Circuit, num_shots: int) -> int:
    sampler = circuit.compile_detector_sampler()
    detection_events, actual_observables = sampler.sample(num_shots, separate_observables=True)
    print("Detection events (first 10):", detection_events[:10])
    print("Non-zero detection events:", np.sum(np.any(detection_events, axis=1)))
    detector_error_model = circuit.detector_error_model(approximate_disjoint_errors=True)

    matcher = BPOSD(detector_error_model, osd_order=0)

    predictions = matcher.decode_batch(detection_events)
    
    num_errors = np.sum(np.any(predictions != actual_observables, axis=1))
    print(num_errors)
    num_vacuous_errors = np.sum(np.any(actual_observables, axis=1))

    print(num_vacuous_errors)

    print(f"{num_errors}/{num_shots}")
    return num_errors / num_shots
    

def generate_pauli_error(p: float) -> PauliNoiseModel:
    pnm = PauliNoiseModel()
    pnm.add_operation("h", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p}) # here the weights do NOT need to be normalized
    #pnm.add_operation("h", {"x": 0.00, "y": 0.00, "z": 0.00,  "i": 1 - 0.000}) # here the weights do NOT need to be normalized
   
    pnm.add_operation("cx", {"ix": p/3, "xi": p/3, "xx": p/3, "ii": 1 - p})
    #pnm.add_operation("id", {"x": 1})
    #pnm.add_operation("reset", {"x": 1})
    pnm.add_operation("measure", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1 - p})
    #pnm.add_operation("x", {"x": p, "y": 0, "z": 0, "i": 1-p})
    return pnm


def try_different_decoder(code):
    H = code.H
    bp_osd = BpLsdDecoder(
            H,
            error_rate = 0.01,
            bp_method = 'product_sum',
            max_iter = 2,
            schedule = 'parallel',
            lsd_method = 'lsd_cs',
            lsd_order = 0
        )
    
    syndrome = np.random.randint(size=H.shape[0], low=0, high=2).astype(np.uint8)

    print(f"Syndrome: {syndrome}")
    decoding = bp_osd.decode(syndrome)
    print(f"Decoding: {decoding}")
    decoding_syndrome = H@decoding % 2
    print(f"Decoding syndrome: {decoding_syndrome}")
    


def run_experiment(experiment_name, backend, code_name, d, num_samples, error_prob):
    code = get_code(code_name, d)
    try:
        detectors, logicals = code.stim_detectors()
        for state, qc in code.circuit.items():
            code.circuit[state] = map_circuit(code.circuit[state], backend)
            gates = set([operation.name for operation in code.circuit[state].data])
            error_model = generate_circuit_specific_pauli_error(gates, error_prob)
            code.noisy_circuit[state] = noisify_circuit(qc, error_model)
            stim_circuit = get_stim_circuits(
                code.noisy_circuit[state], detectors=detectors, logicals=logicals
            )[0][0]
            logical_error_rate = simulate_circuit(stim_circuit, num_samples)
            logging.info(f"{experiment_name} | Logical error rate for {code_name} and state {state} with distance {d}, backend {backend}: {logical_error_rate}")
    
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
        error_prob = experiment["error_probability"]

        parameter_combinations = product(backends, codes, distances)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_experiment, experiment_name, backend, code_name, d, num_samples, error_prob)
                for backend, code_name, d in parameter_combinations
            ]

            for future in futures:
                future.result()

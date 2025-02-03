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


def simulate_circuit(circuit: stim.Circuit, num_shots: int) -> int:
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    #s = circuit.diagram('detslice-with-ops-svg', tick=range(14, 24), filter_coords=['D326', 'D508', 'L0', ])
    #with open("circuit.svg", "w") as f:
    #    f.write(str(s)) # For testing reasons
    circuit.to_file("circuit.stim")
    detector_error_model = circuit.detector_error_model(ignore_decomposition_failures=True,decompose_errors=True, approximate_disjoint_errors=True)

    print("Do we get here?")

    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    print("Does this work?")
    predictions = matcher.decode_batch(detection_events)
    print("Does this work?")
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
    #pnm.add_operation("h", {"x": 0.00, "y": 0.00, "z": 0.00,  "i": 1 - 0.000}) # here the weights do NOT need to be normalized
   
    #pnm.add_operation("cx", {"ix": 1, "xi": 1, "xx": 1})
    #pnm.add_operation("id", {"x": 1})
    #pnm.add_operation("reset", {"x": 1})
    #pnm.add_operation("measure", {"x": 1})
    #pnm.add_operation("x", {"x": p, "y": 0, "z": 0, "i": 1-p})
    return pnm


def run_experiment(experiment_name, backend, code_name, d, num_samples, error_prob):
    try:
        code = get_code(code_name, d)
        detectors, logicals = code.stim_detectors()
        code.circuit['0'] = map_circuit(code.circuit['0'], backend)
        for state, qc in code.circuit.items():
            code.noisy_circuit[state] = noisify_circuit(qc, error_prob)
        
        stim_circuit = get_stim_circuits(
            code.noisy_circuit['0'], detectors=detectors, logicals=logicals
        )[0][0]
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
        error_prob = generate_pauli_error(experiment["error_probability"])

        parameter_combinations = product(backends, codes, distances)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_experiment, experiment_name, backend, code_name, d, num_samples, error_prob)
                for backend, code_name, d in parameter_combinations
            ]

            for future in futures:
                future.result()

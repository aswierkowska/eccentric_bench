import sys
sys.path.append("/home/aswierkowska/eccentric_bench/external/qiskit_qec/src")

import re
import stim
import pymatching
import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, FrozenSet
from scipy.sparse import csc_matrix

import os
import random
import time

from itertools import product
from concurrent.futures import ProcessPoolExecutor
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_qec.circuits import SurfaceCodeCircuit, CSSCodeCircuit, StimCodeCircuit
from qiskit_qec.codes.hhc import HHC
from qiskit_qec.utils import get_stim_circuits, noisify_circuit
from qiskit_qec.noise import PauliNoiseModel
from qiskit_qec.codes.gross_code import GrossCode
from qiskit_qec.codes.bbc import BBCode
from qiskit_qec.circuits.gross_code_circuit import GrossCodeCircuit
from codes_q import create_bivariate_bicycle_codes

from codes import get_gross_code
from noise import add_stim_noise
from BBCODE_DICT import BBCODE_DICT

from custom_backend import FakeLargeBackend

from stimbposd import BPOSD #doesn't work with current ldpcv2 code  pip install -U ldpc==0.1.60
from ldpc import bposd_decoder

def get_code(code_name: str, d: int, depol_error: float = 0.00, bb_tuple=None):
    if code_name == "hh":
        code = HHC(d)
        css_code = CSSCodeCircuit(code, T=d)
        return css_code
    elif code_name == "gross":
        # TODO: should gross code accept parameter?
        return get_gross_code()
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





def dict_to_csc_matrix(elements_dict, shape):
    # Constructs a `scipy.sparse.csc_matrix` check matrix from a dictionary `elements_dict` 
    # giving the indices of nonzero rows in each column.
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)

def dem_to_check_matrices(dem: stim.DetectorErrorModel, return_col_dict=False):

    DL_ids: Dict[str, int] = {} # detectors + logical operators
    L_map: Dict[int, FrozenSet[int]] = {} # logical operators
    priors_dict: Dict[int, float] = {} # for each fault

    def handle_error(prob: float, detectors: List[int], observables: List[int]) -> None:
        dets = frozenset(detectors)
        obs = frozenset(observables)
        key = " ".join([f"D{s}" for s in sorted(dets)] + [f"L{s}" for s in sorted(obs)])

        if key not in DL_ids:
            DL_ids[key] = len(DL_ids)
            priors_dict[DL_ids[key]] = 0.0

        hid = DL_ids[key]
        L_map[hid] = obs
#         priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])
        priors_dict[hid] += prob

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[int] = []
            frames: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    frames.append(t.val)
            handle_error(p, dets, frames)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    check_matrix = dict_to_csc_matrix({v: [int(s[1:]) for s in k.split(" ") if s.startswith("D")] 
                                       for k, v in DL_ids.items()},
                                      shape=(dem.num_detectors, len(DL_ids)))
    observables_matrix = dict_to_csc_matrix(L_map, shape=(dem.num_observables, len(DL_ids)))
    priors = np.zeros(len(DL_ids))
    for i, p in priors_dict.items():
        priors[i] = p

    if return_col_dict:
        return check_matrix, observables_matrix, priors, DL_ids
    return check_matrix, observables_matrix, priors

def modified_bposd_decoder(dem, num_repeat, num_shots, osd_order=10):   
    chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)
    num_row, num_col = chk.shape
    chk_row_wt = np.sum(chk, axis=1)
    chk_col_wt = np.sum(chk, axis=0)
    print(f"check matrix shape {chk.shape}, max (row, column) weight ({np.max(chk_row_wt)}, {np.max(chk_col_wt)}),",
          f"min (row, column) weight ({np.min(chk_row_wt)}, {np.min(chk_col_wt)})")

    bpd = bposd_decoder(
        chk, # the parity check matrix
        channel_probs=priors, # assign error_rate to each qubit. This will override "error_rate" input variable
        max_iter=10000, # the maximum number of iterations for BP
        bp_method="minimum_sum_log", # messages are not clipped, may have numerical issues
        ms_scaling_factor=1.0, # min sum scaling factor. If set to zero the variable scaling factor method is used
        osd_method="osd_cs", # the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
        osd_order=10, # the osd search depth, not specified in [1]
        input_vector_type="syndrome", # "received_vector"
    )

    start_time = time.perf_counter()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    det_data, obs_data, err_data = dem_sampler.sample(shots=num_shots, return_errors=False, bit_packed=False)
    print("detector data shape", det_data.shape)
    print("observable data shape", obs_data.shape)
    end_time = time.perf_counter()
    print(f"Stim: noise sampling for {num_shots} shots, elapsed time:", end_time-start_time)

    num_err = 0
    num_flag_err = 0
    start_time = time.perf_counter()
    for i in range(num_shots):
        e_hat = bpd.decode(det_data[i])
        num_flag_err += ((chk @ e_hat + det_data[i]) % 2).any()
        ans = (obs @ e_hat + obs_data[i]) % 2
        num_err += ans.any()
    end_time = time.perf_counter()
    print("Elapsed time:", end_time-start_time)
    print(f"Flagged Errors: {num_flag_err}/{num_shots}") # expect 0 for OSD
    print(f"Logical Errors: {num_err}/{num_shots}")
    p_l = num_err / num_shots
    p_l_per_round = 1-(1-p_l) ** (1/num_repeat)
    print("Logical error per round:", p_l_per_round)




def simulate_circuit(circuit: stim.Circuit, num_shots: int) -> int:

    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    print(detection_events[:10])
    np.set_printoptions(threshold=sys.maxsize)
    dem = circuit.detector_error_model()
    print("BEFORE DECODER")
    modified_bposd_decoder(dem, 12, num_shots)
    matcher = None

    """
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
    """
    
def generate_pauli_error(p: float) -> PauliNoiseModel:
    pnm = PauliNoiseModel()
    pnm.add_operation("h", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 0}) # here the weights do NOT need to be normalized
    pnm.add_operation("cx", {"ix": p/3, "xi": p/3, "xx": p/3, "ii": 0})
    pnm.add_operation("id", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 0})
    pnm.add_operation("measure", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 0})
    return pnm

def run_experiment(experiment_name, backend, code_name, d, num_samples, error_prob,depol_error=0.00):
    code = get_code(code_name, d,depol_error)
    detectors, logicals = code.detectors, code.logicals
    circuit = code.qc
    #circuit = map_circuit(circuit, backend)
    
    #try:
    #for state, qc in code.circuit.items():
    #    code.noisy_circuit[state] = noisify_circuit(qc, error_prob, depolarizing_prob=error_prob)

    stim_circuit = get_stim_circuits(
        circuit, detectors=detectors, logicals=logicals
    )[0][0]
    stim_circuit = add_stim_noise(stim_circuit, error_prob, error_prob, error_prob, error_prob)
    logical_error_rate = simulate_circuit(stim_circuit, num_samples)
    logging.info(f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend}: {logical_error_rate}")
    
    #except Exception as e:
    #    logging.error(f"{experiment_name} | Failed to run experiment for {code_name}, distance {d}, backend {backend}: {e}")

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
        distances = experiment["distances"]
        depol_error = experiment["depol_error"]
        error_prob = experiment["error_probability"]

        parameter_combinations = product(backends, codes, distances)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_experiment, experiment_name, backend, code_name, d, num_samples, error_prob,depol_error,bb_tuple)
                for backend, code_name, d in parameter_combinations
            ]

            for future in futures:
                future.result()

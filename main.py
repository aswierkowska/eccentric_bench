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
from qiskit_qec.codes.bbc import BBCode
from qiskit_qec.circuits.gross_code_circuit import GrossCodeCircuit
from BBCODE_DICT import BBCODE_DICT

from custom_backend import FakeLargeBackend

from stimbposd import BPOSD#doesn't work with current ldpcv2 code  pip install -U ldpc==0.1.60

#from bposd import bposd_decoder


from qiskit.providers.fake_provider import GenericBackendV2
# from qiskit.visualization import plot_coupling_map
from qiskit.transpiler import CouplingMap
import rustworkx as rx

def generate_cube_map(num_layers, num_rows, num_columns, bidirectional=True):
    """Return a coupling map of qubits connected in a 3D cube structure."""
    def get_index(layer, row, col):
        return layer * (num_rows * num_columns) + row * num_columns + col
    
    graph = rx.PyDiGraph()
    graph.add_nodes_from(range(num_layers * num_rows * num_columns))
    
    for layer in range(num_layers):
        for row in range(num_rows):
            for col in range(num_columns):
                node = get_index(layer, row, col)
                
                if col < num_columns - 1:
                    neighbor = get_index(layer, row, col + 1)
                    graph.add_edge(node, neighbor, None)
                    if bidirectional:
                        graph.add_edge(neighbor, node, None)

                if row < num_rows - 1:
                    neighbor = get_index(layer, row + 1, col)
                    graph.add_edge(node, neighbor, None)
                    if bidirectional:
                        graph.add_edge(neighbor, node, None)
                
                if layer < num_layers - 1:
                    neighbor = get_index(layer + 1, row, col)
                    graph.add_edge(node, neighbor, None)
                    if bidirectional:
                        graph.add_edge(neighbor, node, None)

    return CouplingMap(graph.edge_list(), description="cube")


def get_custom_backend(shape: str, num_qubits: int):
    if shape == "line":
        coupling_map = CouplingMap.from_line(num_qubits)
    elif shape == "grid":
        num_rows, num_cols = int(num_qubits**(1/2)), int(num_qubits**(1/2))
        coupling_map = CouplingMap.from_grid(num_rows, num_cols)
    elif shape == "cube":
        num_layers, num_rows, num_cols = int(num_qubits**(1/3)), int(num_qubits**(1/3)), int(num_qubits**(1/3))
        coupling_map = generate_cube_map(num_layers, num_rows, num_cols)
        # TODO: requires graphviz installed, useful for the paper
        # plot_coupling_map(num_qubits, None, coupling_map.get_edges(), filename="graph.png")
    elif shape == "full":
        coupling_map = CouplingMap.from_full(num_qubits)
    else:
        return None
    
    return GenericBackendV2(coupling_map.size(), coupling_map=coupling_map)














def load_IBM_account():
    load_dotenv()
    token=os.getenv("IBM_TOKEN")
    QiskitRuntimeService.save_account(
    token=token,
    channel="ibm_quantum" # `channel` distinguishes between different account types
    )


def get_code(code_name: str, d: int, depol_error: float = 0.00, bb_tuple=None):
    if code_name == "hh":
        code = HHC(d)
        css_code = CSSCodeCircuit(code, T=d)
        return css_code
    elif code_name == "gross":
        l, m, exp_A, exp_B = BBCODE_DICT[tuple(bb_tuple)]
        code = BBCode(bb_tuple[0],bb_tuple[1],bb_tuple[2],l,m,exp_A,exp_B)
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
        backend = FakeLargeBackend(distance=27, number_of_chips=1)
        circuit = transpile(circuit,  
            backend=backend,
            basis_gates=stim_gates,
            optimization_level=0
        )
    else:
        try:
            backend = get_custom_backend("full", 288)
            circuit = transpile(circuit, 
                backend=backend,
                basis_gates=stim_gates,
                optimization_level=0
            )
        except:
            raise ValueError(f"Backend {backend} not supported) ")
    return circuit


def simulate_circuit(circuit: stim.Circuit, num_shots: int, code=None) -> float:
    sampler = circuit.compile_detector_sampler()
    print("Smapler Done")
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    #Log obeservable flips
    print(len(detection_events[0]))
    print("Detection Events Done")
    np.set_printoptions(threshold=sys.maxsize)
    detector_error_model = circuit.detector_error_model(approximate_disjoint_errors=True,
                                                         allow_gauge_detectors=True,
                                                         flatten_loops=True, decompose_errors=True,
                                                         ignore_decomposition_failures=True,
                                                         block_decomposition_from_introducing_remnant_edges=True)
    print("Error Model Done")
    #VERSION 3 Pymatching

    #matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    #predictions = matcher.decode_batch(detection_events)
    #num_errors = 0
    #for shot in range(num_shots):
    #    actual_for_shot = observable_flips[shot]
    #    predicted_for_shot = predictions[shot]
    #    if not np.array_equal(actual_for_shot, predicted_for_shot):
    #        num_errors += 1
    #return num_errors / num_shots



    #VERSION 2 BPOSD DECODER
    #print(observable_flips)
    #bpd = bposd_decoder(
    #    code.H,#the parity check matrix
    #    error_rate=0.05,
    #    channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
    #    max_iter=10, #the maximum number of iterations for BP)
    #    bp_method="ms",
    #    ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
    #    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    #    osd_order=7 #the osd search depth
    #)
    #sum = 0
    #for detection_event in detection_events:
    #    #turn True into 1 and False into 0
    #    detection_event = [int(i) for i in detection_event]
    #    #print(len(detection_event))
    #    syndrome = code.H@detection_event % 2
    #    bpd.decode(syndrome)
    #    residual_error = (bpd.osdw_decoding+detection_event) % 2
    #    #print(bpd.osdw_decoding)
    #    #print(code.logical_x)
    #    logicals = code.logical_x + code.logical_z
    #    a = (logicals@residual_error%2).any() 
    #    if a: sum+=1
    #
    #return sum/num_shots


    #VERSION 1 BPOSD DECODER
    matcher = BPOSD(detector_error_model, bp_method="min_sum", max_bp_iters=144, osd_order=10 ,osd_method="osd_cs")
    #print("Matcher Done")
    predictions = []
    for detection_event in detection_events:
        prediction = matcher.decode(detection_event)
        predictions.append(prediction)
    
    print("Predictions Done")

    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors / num_shots
    #return 0.0

def generate_pauli_error(p: float) -> PauliNoiseModel:
    pnm = PauliNoiseModel()
    pnm.add_operation("h", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1-p}) # here the weights do NOT need to be normalized
    pnm.add_operation("cx", {"ix": p/3, "xi": p/3, "xx": p/3, "ii": 1-p})
    pnm.add_operation("id", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1-p})
    pnm.add_operation("measure", {"x": p / 3, "y": p / 3, "z": p / 3, "i": 1-p})
    print("Noise Added")
    return pnm


def run_experiment(experiment_name, backend, code_name, d, num_samples, error_prob,depol_error=0.00,bb_tuple=None):
    code = get_code(code_name, d,depol_error,bb_tuple=bb_tuple)
    detectors, logicals = code.stim_detectors()
    code.circuit['0'] = map_circuit(code.circuit['0'], backend)
    try:
        for state, qc in code.circuit.items():
            code.noisy_circuit[state] = noisify_circuit(qc, error_prob)
        
        #circuit to pdf
        #code.noisy_circuit['0'].draw(output='mpl', filename='circuit.pdf',vertical_compression='high', scale=0.3, fold=500)
        s = get_stim_circuits(
            code.noisy_circuit['0'], detectors=detectors, logicals=logicals
        )
        stim_circuit = s[0][0]
        #print(s[1])
        #TODO ADD INSERT FUNCTION TO ADD DEPOL NOISE
        #stim_circuit.safe_insert("What are the paramesters")

        #stim_circuit.to_file(f"{experiment_name}_{code_name}_{d}_{backend}.stim")
        #add depol noise to circuit
        #with open(f"{experiment_name}_{code_name}_{d}_{backend}.stim", "r") as f:
        #    stim_circuit = f.readlines()
        #    stim_circuit.insert(1, f"DEPOLARIZE1(0.1) 1 3 5 8 10 12\n")
        #    #write stim circuit back to file
        #    with open(f"{experiment_name}_{code_name}_{d}_{backend}.stim", "w") as f:
        #        f.writelines(stim_circuit)

        #stim_circuit = stim.Circuit.from_file(f"{experiment_name}_{code_name}_{d}_{backend}.stim")



        #print(stim_circuit)
        logical_error_rate = simulate_circuit(stim_circuit, num_samples, code)
        logging.info(f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend}: {logical_error_rate:.4f}")
    
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
        bb_tuple = experiment["bb_tuple"]
        print(error_prob.to_dict())

        parameter_combinations = product(backends, codes, distances)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_experiment, experiment_name, backend, code_name, d, num_samples, error_prob,depol_error,bb_tuple)
                for backend, code_name, d in parameter_combinations
            ]

            for future in futures:
                future.result()

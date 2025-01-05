import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt
from qiskit_qec.utils import get_stim_circuits, noisify_circuit
from qiskit_qec.noise import PauliNoiseModel
from qiskit import QuantumCircuit, QuantumRegister
from mqt.qecc.codes import CSSCode
from mqt.qecc.ft_stateprep import gate_optimal_prep_circuit
from mqt.qecc.ft_stateprep import NoisyNDFTStatePrepSimulator
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_qec.circuits import SurfaceCodeCircuit, CSSCodeCircuit
from qiskit_qec.decoders.decoding_graph import DecodingGraph
from qiskit_qec.codes.codebuilders import SurfaceCodeBuilder
from qiskit_qec.codes.hhc import HHC

#def get_code(code: str, d: int):
#    surface_code = CSSCode.from_code_name("surface", 3)
#    ft_prep = gate_optimal_prep_circuit(surface_code, zero_state=True, max_timeout=2)
#    return ft_prep.circ

def get_code(code_name: str, d: int, state: int):
    pnm = PauliNoiseModel()
    p = 0.002
    pnm.add_operation("h", {"x": p, "y": p, "z": p})
    #pnm.add_operation("cx", {"ix": 1, "xi": 1, "xx": 1})
    #pnm.add_operation("special", {"x": 1, "y": 1})
    #pnm.add_operation("measure", {"x": 1})
    #pnm.add_operation("reset", {"x": 1})
    #pnm.set_error_probability("h", p)
    #pnm.set_error_probability("cx", p)
    #pnm.set_error_probability("special", p)
    #pnm.set_error_probability("measure", p)
    #pnm.set_error_probability("reset", p)
    error_rate = 0.002
    noise_model = (error_rate, error_rate)
    error_rate = 0.002
    if code_name == "hh":
        code = HHC(d)
        #css_code = CSSCodeCircuit(code, T=d, noise_model=pnm) # noise_model=(error_rate, error_rate)
        css_code = CSSCodeCircuit(code, T=d)
        return css_code
    elif code_name == "surface":
        #code = SurfaceCodeBuilder(d=d).build()
        code = SurfaceCodeCircuit(d=d, T=d)
    


def map_circuit(code: CSSCode, backend: str):
    if backend[:3] == "ibm":
        service = QiskitRuntimeService(instance="ibm-q/open/main")
        backend = service.backend(backend)
        
        code.circuit["0"] = transpile(code.circuit["0"], 
            backend=backend,
            # basis_gates=['h', 'cx', 's', 'x', 'z', 's_dag', 'swap'], # TODO: update with gates available in stim   circuit: Compatible gates are Paulis, controlled Paulis, h, s, and sdg, swap, reset, measure and barrier
            basis_gates=['x', 'y', 'z', 'cx', 'cz', 'cy', 'h', 's', 's_dag', 'swap', 'reset', 'measure', 'barrier'],
            optimization_level=0
        )
        print(code.circuit["0"])
        code.circuit["1"] = transpile(code.circuit["1"], 
            backend=backend,
            # basis_gates=['h', 'cx', 's', 'x', 'z', 's_dag', 'swap'], # TODO: update with gates available in stim   circuit: Compatible gates are Paulis, controlled Paulis, h, s, and sdg, swap, reset, measure and barrier
            basis_gates=['x', 'y', 'z', 'cx', 'cz', 'cy', 'h', 's', 's_dag', 'swap', 'reset', 'measure', 'barrier'],
            optimization_level=0
        )
    return code

def to_stim(circuit: QuantumCircuit):
    return circuit

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

if __name__ == '__main__':
    # parameters:
    num_samples = 10000 
    backends = ["ibm_brisbane"]
    codes = ["hh",] # "surface"
    distances = [3]
    pnm = PauliNoiseModel()
    p = 0.0001
    #pnm.add_operation("h", {"x": 1, "y": 1, "z": 1})
    #pnm.add_operation("cx", {"ix": 1, "xi": 1, "xx": 1})
    #pnm.add_operation("special", {"x": 1, "y": 1})
    #pnm.add_operation("measure", {"x": 1})
    #pnm.add_operation("reset", {"x": 1})
    #pnm.set_error_probability("h", p)
    #pnm.set_error_probability("cx", p)
    #pnm.set_error_probability("special", p)
    #pnm.set_error_probability("measure", p)
    #pnm.set_error_probability("reset", p)
    error_rate = 0.002
    noise_model = (error_rate, error_rate)

    # topology experiment
    for backend in backends:
        for code_name in codes:
            for d in distances:
                code = get_code(code_name, d, state=0)
                code = map_circuit(code, backend)
                detectors, logicals = code.stim_detectors()
                print(logicals)
                print(detectors)
                #graph = DecodingGraph(code).graph
                #m = pymatching.Matching(graph)
                #code.noisy_circuit = noisify_circuit(code.circuit.values(), noise_model)

                #print(circuit)
                # logicals contains final_readouts
                #print(code.noisy_circuit["0"])
                stim_circuit = get_stim_circuits(
                     code.circuit["0"], detectors=detectors, logicals=logicals
                )[0][0]
                print(stim_circuit)
                #print(stim_circuit_clean)
                #print(stim_circuit)
                
                logical_error_rate = simulate_circuit(stim_circuit, num_samples)
                print(logical_error_rate)
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
from qiskit_aer.noise import depolarizing_error, pauli_error
from qiskit.circuit import Measure

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
    error_rate = 0.002
    pnm.add_operation("h", {"x": 1, "y": 1, "z": 1})
    pnm.add_operation("cx", {"ix": 1, "xi": 1, "xx": 1})
    pnm.add_operation("special", {"x": 1, "y": 1})
    pnm.add_operation("measure", {"x": 1})
    pnm.add_operation("reset", {"x": 1})
    pnm.set_error_probability("h", error_rate)
    pnm.set_error_probability("cx", error_rate)
    pnm.set_error_probability("special", error_rate)
    pnm.set_error_probability("measure", error_rate)
    pnm.set_error_probability("reset", error_rate)
    
    noise_model = (error_rate, error_rate)
    d = 3
    code = HHC(d)
    css_code = CSSCodeCircuit(code, T=d)
    for state, qc in css_code.circuit.items():
        noisy_qc = qc.copy()  # Start with a copy of the original circuit

    # Define noise channels
    depol_error = depolarizing_error(error_rate, 1)
    meas_error = pauli_error([("X", error_rate), ("I", 1 - error_rate)])

    css_code.noisy_circuit = {}


    for qubit in range(css_code.code.n):  # Loop over qubits
        for instruction in noisy_qc.data:
                operation = instruction.operation 
                qubits = instruction.qubits    
                clbits = instruction.clbits   
                if isinstance(operation, Measure):
                    noisy_qc.append(meas_error.to_instruction(), [qubits[0]])
                #else:
                # depolaryzacja wywala vscode, musimy doczytać kiedy powinna być realizowana
                #    noisy_qc.append(depol_error.to_instruction(), [qubits[0]])
                
        # Add depolarizing noise between rounds (if needed)
        #for qubit in range(css_code.code.n):  # Apply to all code qubits
        #    noisy_qc.append(depol_error.to_instruction(), [qubit])

        # Tutaj musimy zdefiniować do jakiego stanu to ma być przypisane
        #css_code.noisy_circuit[state] = noisy_qc

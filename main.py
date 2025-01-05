import stim
import cirq
import cirq_google
import stimcirq
#from iqm import cirq_iqm

import pymatching
import numpy as np
import matplotlib.pyplot as plt
from qiskit_qec.utils import get_stim_circuits, noisify_circuit
from qiskit_qec.noise import PauliNoiseModel
from qiskit import QuantumCircuit, QuantumRegister
from mqt.qecc.codes import CSSCode
from mqt.qecc.ft_stateprep import gate_optimal_prep_circuit
from utils import get_qiskit_circuits
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

# codes = ["color_code:memory_xyz", "surface_code:rotated_memory_x"]
codes = ["surface_code:rotated_memory_x"]

 
# Function to count logical errors given a circuit and number of shots
def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    print(detector_error_model)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors

# Function to simulate surface code and calculate logical error rate
def simulate_surface_code_with_pymatching(code, distance, rounds, noise, num_samples): # I should pass code, dist, noise model
    # Generate the surface code circuit
    circuit = stim.Circuit.generated(
        code,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization = 0.2
    )

    #print(circuit)

    surface_code = CSSCode.from_code_name("surface", 3)
    ft_sp = gate_optimal_prep_circuit(surface_code, zero_state=True, max_timeout=2)

    service = QiskitRuntimeService(instance="ibm-q/open/main")
    backend = service.backend('ibm_brisbane')
    # print(backend.configuration().coupling_map)
    circuit = transpile(
        ft_sp.circ,
        backend=backend,
        basis_gates=['h', 'cx', 's', 'x', 'z', 's_dag', 'swap'],
        optimization_level=0,
        #routing_method='sabre',
    #initial_layout=[0, 1, 2],  # Map logical qubits to physical qubits
    ) 
    
    
    #print(stim_circ)    

    #print(circuit)

    #qiskit_circuit = get_qiskit_circuits(circuit)
    #cirq_circuit = stimcirq.stim_circuit_to_cirq_circuit(circuit)
    #decomposed_circuit = cirq.decompose(cirq_circuit)
    #if isinstance(decomposed_circuit, list):
    #    decomposed_circuit = cirq.Circuit(decomposed_circuit)
    #device = cirq_google.Sycamore
    #device_graph = device.metadata.nx_graph
    #router = cirq.RouteCQC(device_graph)
    #routed_circuit = router(decomposed_circuit)

    #sycamore_gateset = cirq_google.SycamoreTargetGateset()
    #optimized_circuit = cirq.optimize_for_target_gateset(routed_circuit, gateset=sycamore_gateset)
    #stim_compatible_circuit = cirq.Circuit()
    #for moment in optimized_circuit:
    #    for op in moment:
    #        # Attempt to decompose each operation
    #        try:
    #            stim_compatible_circuit.append(cirq.decompose(op))
    #        except TypeError:
    #            # If the gate is already compatible or decomposition fails, keep as is
    #            stim_compatible_circuit.append(op)

    # Convert to stim
    #stim_circuit = stimcirq.cirq_circuit_to_stim_circuit(stim_compatible_circuit)

    #routed_circuit = cirq.optimize_for_target_gateset(cirq_circuit, gateset = cirq_google.SycamoreTargetGateset())
    # Google device does not support reset
    # Validate our circuit
    #device.validate_circuit(routed_circuit)

    #exit(0)


    # TODO: nosify_circuit from qiskit
    #noise_model = NoiseModel.from_backend(backend)
    #print(noise_model)

    pnm = PauliNoiseModel()
    p = 0.2
    pnm.add_operation("h", {"x": 1, "y": 1, "z": 1})
    pnm.add_operation("cx", {"ix": 1, "xi": 1, "xx": 1})
    pnm.add_operation("special", {"x": 1, "y": 1})
    pnm.add_operation("measure", {"x": 1})
    pnm.add_operation("reset", {"x": 1})
    pnm.set_error_probability("h", p)
    pnm.set_error_probability("cx", p)
    pnm.set_error_probability("special", p)
    pnm.set_error_probability("measure", 0.2)
    pnm.set_error_probability("reset", 0.2)cp
    circuit = noisify_circuit(circuit, pnm)
    #print(circuit)
    circuit = get_stim_circuits(circuit)[0][0]
    #print(circuit)

    # Count logical errors using the count_logical_errors function
    num_errors = count_logical_errors(circuit, num_samples)
    
    # Calculate logical error rate
    logical_error_rate = num_errors / num_samples
    return logical_error_rate

# Simulation parameters
distances = [3]  # Code distances to test
rounds = 10               # Number of stabilizer rounds
noise = 0.001             # Depolarizing noise probability
num_samples = 10000       # Number of trials per simulation

qreg = QuantumRegister(2, "q")
qiskit_circuit = QuantumCircuit(qreg)
qiskit_circuit.x(qreg[0])
stim_circuit = get_stim_circuits(qiskit_circuit)

# Run simulations for each distance and collect results
results  = {}

for code in codes:
    results[code] = []
    for d in distances:
        logical_error_rate = simulate_surface_code_with_pymatching(code, d, rounds, noise, num_samples)
        results[code].append((d, logical_error_rate))
        print(f"Distance {d}, Logical Error Rate: {logical_error_rate}")

# Extract results for plotting
"""
distances, error_rates = zip(*results)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(distances, error_rates, marker='o', linestyle='-', color='b', label='Surface Code')
plt.yscale('log')  # Logical error rates are typically shown on a log scale
plt.xlabel("Code Distance")
plt.ylabel("Logical Error Rate")
plt.title("Error Suppression in Surface Code with PyMatching")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("result.png")

"""
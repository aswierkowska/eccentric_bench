from qiskit.transpiler import Target, InstructionProperties, CouplingMap
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeKyiv
from qiskit.providers import QubitProperties
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import InstructionDurations

backend = FakeKyiv()
durations = InstructionDurations.from_backend(backend)

# Optionally convert to seconds using dt value
dt = backend.configuration().dt
print(f"\nBackend dt: {dt} seconds")

print("\nReset gate durations (in seconds):")
for qubit in range(backend.configuration().num_qubits):
    duration = durations.get("reset", [qubit])
    duration_sec = duration * dt if duration else None
    print(f"Qubit {qubit}: {duration_sec} s")
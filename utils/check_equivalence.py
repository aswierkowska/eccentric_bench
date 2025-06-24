from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import sys
import os
from qiskit.converters import circuit_to_dag

def are_structurally_equivalent(circ1, circ2):
    try:
        dag1 = circuit_to_dag(circ1)
        dag2 = circuit_to_dag(circ2)
        return dag1 == dag2
    except Exception as e:
        print(f"Error comparing DAGs: {e}")
        return False

def load_qasm_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return QuantumCircuit.from_qasm_file(path)
    except Exception as e:
        raise ValueError(f"Failed to load QASM file {path}: {e}")

def main(file1, file2):
    circ1 = load_qasm_file(file1)
    circ2 = load_qasm_file(file2)

    if are_structurally_equivalent(circ1, circ2):
        print("The circuits are equivalent (up to global phase).")
    else:
        print("The circuits are NOT equivalent.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_qasm_equivalence.py <file1.qasm> <file2.qasm>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])

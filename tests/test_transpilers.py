import pytest
from qiskit import QuantumCircuit
from bqskit.ir.gates import *
from pytket import OpType
from transpilers import translate

qiskit_simplified_gates = ["rz", "cx", "h"]
bqskit_simplified_gates = [RZGate(), CXGate(), HGate()]
tket_simplified_gates = [OpType.Rz, OpType.CX, OpType.H]

def decomposable_circuit():
    qc = QuantumCircuit(3)
    qc.u(0.5, 1.2, 2.3, 0)
    qc.t(1)
    qc.swap(0, 2)
    qc.ccx(0, 1, 2)
    return qc

def get_gate_names(qc):
    return {inst.operation.name.lower() for inst in qc.data}

@pytest.mark.parametrize("method,expected_gates", [
    ("qiskit", qiskit_simplified_gates),
    ("bqskit", bqskit_simplified_gates),
    ("tket", tket_simplified_gates),
])
def test_translate_with_decomposition(method, expected_gates):
    qc = decomposable_circuit()
    translated = translate(qc, method, gate_set=expected_gates)
    used_gates = get_gate_names(translated)
    if method == "qiskit":
        expected_gate_names = {str(gate).lower() for gate in expected_gates}
    elif method == "bqskit":
        expected_gate_names = {str(gate).lower().replace('gate', '').replace('cnot', 'cx') for gate in expected_gates}
    elif method == "tket":
        expected_gate_names = {gate.name.lower() for gate in expected_gates}
    unexpected_gates = used_gates - expected_gate_names
    assert not unexpected_gates, f"Unexpected gates in {method} translation: {unexpected_gates}"

def test_translate_invalid_method():
    qc = QuantumCircuit(1)
    with pytest.raises(NotImplementedError):
        translate(qc, "unknown_method")

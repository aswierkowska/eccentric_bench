import logging
from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit
from bqskit import MachineModel, compile
from bqskit.ir.gates import *
from pytket import OpType
from pytket.passes import AutoRebase
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import transpile


qiskit_stim_gates = [
    "x",
    "y",
    "z",
    "cx",
    "cz",
    "cy",
    "h",
    "s",
    "s_dag",
    "swap",
    "reset",
    "measure",
    "barrier",
]

bqskit_stim_gates = [
    XGate(),
    YGate(),
    ZGate(),
    CXGate(),
    CZGate(),
    CYGate(),
    HGate(),
    SGate(),
    SdgGate(),
    SwapGate(),
    Reset(),
    #MeasureGate(),
    #BarrierPlaceholder(), # it deals with measure/barrier anyway it seems
]

# TODO: either we need to provide custom tk1 to clifford gates decomposition or use a diff set for experiments
tket_stim_gates = [
    OpType.X,
    OpType.Y,
    OpType.Z,
    OpType.CX,
    OpType.CZ,
    OpType.CY,
    OpType.H,
    OpType.S,
    OpType.Sdg,
    OpType.SWAP,
    OpType.Reset,
    OpType.Measure,
    OpType.Barrier,
    # TODO: remove
    #OpType.Rz
]

#TODO: do we want to add ID?
qiskit_ibm_heron = [
    "x",
    "sx",
    "rzz",
    "rz",
    "rx",
    'cz',
    'measure',
    'reset',
    'barrier',
]

bqskit_ibm_heron = [
    XGate(),
    SXGate(),
    RZZGate(),
    RZGate(),
    RXGate(),
    CZGate(),
    Reset(),

]

tket_ibm_heron = [
    OpType.X,
    OpType.SX,
    OpType.ZZPhase, #same as RZZ
    OpType.Rz,
    OpType.Rx,
    OpType.CZ,
    OpType.Measure,
    OpType.Reset,
    OpType.Barrier,
]


def translate(circuit, translating_method, gate_set=None):
    if translating_method == "qiskit":
        qiskit_circuit = transpile(
            circuit,
            basis_gates=gate_set if gate_set else qiskit_stim_gates,
            optimization_level=0,
        )
        return qiskit_circuit
    elif translating_method == "bqskit":
        # TODO: bqskit renames which later causes ['round_0_zplaq_bit', 0] is not in list
        bqskit_circuit = qiskit_to_bqskit(circuit)
        model = MachineModel(bqskit_circuit.num_qudits, gate_set=gate_set if gate_set else bqskit_stim_gates)
        # BQSkit doesn't allow for 0 level of optimizations
        bqskit_circuit = compile(bqskit_circuit, model=model, optimization_level=1)
        return bqskit_to_qiskit(bqskit_circuit)
    elif translating_method == "tket":
        tket_circuit = qiskit_to_tk(circuit)
        rebase_pass = AutoRebase(set(gate_set if gate_set else tket_stim_gates))
        rebase_pass.apply(tket_circuit)
        return tk_to_qiskit(tket_circuit)
    else:
        logging.error(f'Unknown translate method: {translating_method}')
        raise NotImplementedError
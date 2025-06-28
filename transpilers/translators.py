import logging
from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit
from bqskit import MachineModel, compile
from bqskit.ir.gates import *
from pytket import OpType
from pytket.passes import AutoRebase
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import transpile


qiskit_stim = [
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

bqskit_stim = [
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
tket_stim = [
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

#qiskit has no fixed zz gate
#added rxx and ryy gates to simulate SU(4)
qiskit_h2 = [
    "rx",
    "ry",
    "rz",
    "rzz",
    'rxx',
    "ryy"
    "reset",
    "measure",
    "barrier",
]

bqskit_h2 = [
    RXGate(),
    RYGate(),
    RZGate(),
    ZZGate(),
    RZZGate(),
    RXXGate(),
    RYYGate(),
    Reset(),
]



tket_h2 =[
    OpType.Rx,
    OpType.Ry,
    OpType.Rz,
    OpType.ZZPhase,
    OpType.ZZMax,
    OpType.TK2,
    OpType.Measure,
    OpType.Reset,
    OpType.Barrier,
]


def translate(circuit, translating_method, gate_set=None):
    if gate_set:
        full_gate_set_name = f"{translating_method}_{gate_set}"
        try:
            gate_set_obj = globals()[full_gate_set_name]
        except KeyError:
            logging.error(f"Gate set '{full_gate_set_name}' is not defined.")
            raise ValueError(f"Unknown gate set: {full_gate_set_name}")
    else:
        gate_set_obj = {
            "qiskit": qiskit_stim,
            "bqskit": bqskit_stim,
            "tket": tket_stim,
        }.get(translating_method)

        if gate_set_obj is None:
            logging.error(f"Default gate set not defined for: {translating_method}")
            raise ValueError(f"Missing default gate set for: {translating_method}")

    if translating_method == "qiskit":
        qiskit_circuit = transpile(
            circuit,
            basis_gates=gate_set_obj,
            optimization_level=0,
        )
        return qiskit_circuit

    elif translating_method == "bqskit":
        bqskit_circuit = qiskit_to_bqskit(circuit)
        model = MachineModel(bqskit_circuit.num_qudits, gate_set=gate_set_obj)
        print(model.gate_set)
        bqskit_circuit = compile(bqskit_circuit, model=model, optimization_level=1)
        #return bqskit_to_qiskit(bqskit_circuit)
        return bqskit_circuit

    elif translating_method == "tket":
        tket_circuit = qiskit_to_tk(circuit)
        rebase_pass = AutoRebase(set(gate_set_obj))
        rebase_pass.apply(tket_circuit)
        #return tk_to_qiskit(tket_circuit)
        return tket_circuit

    else:
        logging.error(f"Unknown translate method: {translating_method}")
        raise NotImplementedError

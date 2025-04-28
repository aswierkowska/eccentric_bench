import stim
from utils.math import *

from qiskit_qec.circuits import StimCodeCircuit

def get_surface_code(d, T = None):
    if T == None:
        T = d
    stim_circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
        rounds=T,
        distance=d
    )
    return StimCodeCircuit(stim_circuit = stim_circuit)
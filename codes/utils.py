import math
from qiskit_qec.circuits import SurfaceCodeCircuit, CSSCodeCircuit
from qiskit_qec.codes.hhc import HHC
from .gross_code import get_gross_code
from .color_code_stim import get_color_code
from .bacon_shor import get_bacon_shot_code
from .concat_steane import get_concat_steane_code
from .surface_code import get_surface_code
from .hh_code import get_hh_code

def get_code(code_name: str, d: int, cycles: int):
    if code_name == "hh":
        if cycles == None:
            return get_hh_code(d, None)
        else:
             return get_hh_code(d, cycles)
    elif code_name == "gross":
        if cycles == None:
        # TODO: should gross code accept parameter?
            return get_gross_code()
        else:
            return get_gross_code(T=cycles)
    elif code_name == "surface":
        if cycles == None:
            code = get_surface_code(d=d, T=d)
        else:
            code = get_surface_code(d=d, T=cycles)
        return code
    elif code_name == "color":
        if cycles == None:
            return get_color_code(d, rounds=d)
        else:
            return get_color_code(d, rounds=cycles)
    elif code_name == "bacon":
        if cycles == None:
            return get_bacon_shot_code(d)
        else:
            return get_bacon_shot_code(d, cycles)
    elif code_name == 'steane':
        if d == 3:
            m = 1
        elif d == 9:
            m = 2
        elif d == 27:
            m = 3
        else:
            raise ValueError("Steane code only supports m = 1, 2, 3")
        return get_concat_steane_code(m)


def get_max_d(code_name: str, n: int):
    if code_name == "surface":
        # d**2 data qubits + d**2 - 1 ancilla qubits
        # TODO: tmp solution
        #d = math.floor(math.sqrt((n + 1) / 2))
        #d = d - ((1 - d) % 2)
        if n >= 494:
            return 15
        elif n >= 376:
            return 13
        elif n >= 250: # TODO Arbitrary
            return 11         
    elif code_name == "hh":
        # n = 5d^2 - 2d - 1 /2
        d = int((2 + math.sqrt(40 * n + 24)) / 10)
        d = d - ((1 - d) % 2)
        return d
    elif code_name == "gross":
        return math.floor(n / 2)
    elif code_name == "color":
        d = int((math.sqrt(4*n) +1)/3)
        d = d - ((1 - d) % 2)
        return d
    elif code_name == "bacon":
        #TODO: check this
        #assuming square lattice n = d^2
        #actually it should be d = min(m,n) according to qecc zoo but we have no m,n structure
        d = int(math.sqrt(n))
        d = d - ((1 - d) % 2) 
        return d
    elif code_name == 'steane':
        if n >= 686: # According to Stim file
            return 27
        elif n >= 98: # According to Stim file
            return 9
        elif n >= 13:
            return 3
    return 0

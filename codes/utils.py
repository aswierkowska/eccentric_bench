import math
from qiskit_qec.circuits import SurfaceCodeCircuit, CSSCodeCircuit
from qiskit_qec.codes.hhc import HHC
from .gross_code import get_gross_code
from .color_code_stim import get_color_code

def get_code(code_name: str, d: int):
    if code_name == "hh":
        code = HHC(d)
        css_code = CSSCodeCircuit(code, T=d)
        return css_code
    elif code_name == "gross":
        # TODO: should gross code accept parameter?
        return get_gross_code()
    elif code_name == "surface":
        code = SurfaceCodeCircuit(d=d, T=d)
        return code
    elif code_name == "color":
        return get_color_code(d)


def get_max_d(code_name: str, n: int):
    if code_name == "surface":
        # d**2 data qubits + d**2 - 1 ancilla qubits
        d = math.floor(math.sqrt((n + 1) / 2))
        d = d - ((1 - d) % 2)
        return d
    elif code_name == "hh":
        # n = 5d^2 - 2d - 1 /2
        d = int((2 + math.sqrt(40 * n + 24)) / 10)
        d = d - ((1 - d) % 2)
        return d
    elif code_name == "gross":
        return math.floor(n / 2)
    elif code_name == "color":
        #TODO check actually which distance to put here
        d = int((math.sqrt(4*n) +1)/3)
        d = d - ((1 - d) % 2)
        print("Distance for color code: ", d)
        return d
    return 0

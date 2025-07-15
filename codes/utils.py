import math
from .gross_code import get_gross_code
from .color_code_stim import get_color_code
from .bacon_shor import get_bacon_shor_code
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
            return get_bacon_shor_code(d)
        else:
            return get_bacon_shor_code(d, cycles)
    elif code_name == 'steane':
        if d == 3:
            m = 1
        elif d == 9:
            m = 2
        elif d == 27:
            m = 3
        else:
            raise ValueError("Steane code only supports m = 1, 2, 3")
        if cycles == None:
            cycles = d
        return get_concat_steane_code(m,cycles, basis="Z")


def get_max_d(code_name: str, n: int):
    if code_name == "surface":
        if n < 26:
            raise 1 #Note that a Error will be logged in the main.py

        d = int((-3 + math.isqrt(9 + 8*(n+1))) // 4)
        d = d - ((1 - d) % 2)
        assert d % 2 == 1
        return d
    elif code_name == "hh":
        # n = 5d^2 - 2d - 1 /2
        d = int((2 + math.sqrt(40 * n + 24)) / 10)
        d = d - ((1 - d) % 2) # In order to mamize the logical error rate we want odd distances according to the heavy hex author: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.011022
        assert d % 2 == 1
        return d
    elif code_name == "gross":
        return 12
    elif code_name == "color":
        d = int((math.sqrt(4*n) +1)/3)
        d = d - ((1 - d) % 2) # we look into triangular color code which has odd distance
        assert d % 2 == 1
        return d
    elif code_name == "bacon":
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


def get_min_n(code_name: str, d: int):
    if code_name == "surface":
        return 2 * d * d + 3 * d - 1
    elif code_name == "hh":
        return int(math.ceil((5 * d * d - 2 * d - 1) / 2))
    elif code_name == "gross":
        return 288
    elif code_name == "color":
        return int(math.ceil((3*d - 1)**2 / 4))
    elif code_name == "bacon":
        return d * d #this is only partially true d*d is for sure the lower bound but code actually might require more qubits
    elif code_name == 'steane':
        if d == 3:
            return 14
        elif d == 9:
            return 98
        elif d == 27:
            return 686
    return 0

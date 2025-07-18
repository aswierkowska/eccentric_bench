import sys
import os

#TODO fix this
# Dynamically add the src directory to the sys.path
external_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './../external/more-bacon-less-threshold/src'))
if external_src_path not in sys.path:
    sys.path.append(external_src_path)

# Import the module
from baconshor import _bacon_shor
from baconshor._bacon_shor import make_bacon_shor_constructions
from gen import _gen_util

from qiskit_qec.circuits import StimCodeCircuit

def get_bacon_shor_code(d=5, cycles=1):
    #This custom value should be fixed because it seems there lies the problem
    #Workflow:
    #https://github.com/Strilanc/more-bacon-less-threshold/blob/main/step1_generate_circuits.sh
    #https://github.com/Strilanc/more-bacon-less-threshold/blob/main/tools/gen_circuits
    #https://github.com/Strilanc/more-bacon-less-threshold/blob/main/src/gen/_gen_util.py
    #https://github.com/Strilanc/more-bacon-less-threshold/blob/main/src/baconshor/_bacon_shor.py


    """Added Comments because the original code does not have many comments

        Args:
            d (int): Distance of the code
            custom (dict): Custom parameters for the circuit, 'b' is the basis in which should be measured 'X' or 'Z'
            diameter (int): Diameter of the circuit, just the distance
            rounds (int): Number of measurement cycles of the circuit
            style (str): Style of the circuit, 'bacon_shor' is set per default there are other bacon shor versions we do not use
            
    
    """

    construction = {**make_bacon_shor_constructions()}
    circuit = _gen_util._generate_single_circuit(
                constructions=construction,
                params=_gen_util.CircuitBuildParams(style='bacon_shor', rounds=cycles, diameter=d, custom={'b': 'Z'}),
                noise=None,
                debug_out_dir=None,
                convert_to_cz='0'
            )
    #print(circuit)
    return StimCodeCircuit(stim_circuit=circuit)


if __name__ == "__main__":
    get_bacon_shor_code()

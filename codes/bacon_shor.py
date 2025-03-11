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

print(dir(_bacon_shor))

def get_bacon_shot_code(d=5):
    #This custom value should be fixed because it seems there lies the problem
    #Workflow:
    #https://github.com/Strilanc/more-bacon-less-threshold/blob/main/step1_generate_circuits.sh
    #https://github.com/Strilanc/more-bacon-less-threshold/blob/main/tools/gen_circuits
    #https://github.com/Strilanc/more-bacon-less-threshold/blob/main/src/gen/_gen_util.py
    #https://github.com/Strilanc/more-bacon-less-threshold/blob/main/src/baconshor/_bacon_shor.py

    construction = {**make_bacon_shor_constructions()}
    circuit = _gen_util._generate_single_circuit(
                constructions=construction,
                params=_gen_util.CircuitBuildParams(style='bacon_shor', rounds=5, diameter=d, custom={'b': 'X'}),
                noise=None,
                debug_out_dir=None,
                convert_to_cz='auto',
            )
    return StimCodeCircuit(stim_circuit=circuit)

    #we should get stim circuit from that and the we can just return it

if __name__ == "__main__":
    get_bacon_shot_code()
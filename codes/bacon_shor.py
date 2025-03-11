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

print(dir(_bacon_shor))

def get_bacon_shot_code():
    #_gen_util.py -> _generate_single_circuit retuns stim circuit then we can do the same wrapper i guess

    #_bacon_shor.make_bacon_shor_constructions
    construction = {**make_bacon_shor_constructions()}

    circuit = _gen_util._generate_single_circuit(
                constructions=construction,
                params=_gen_util.CircuitBuildParams(style='bacon_shor', rounds=5, diameter=5, custom={'b': 1}),
                noise=None,
                debug_out_dir=None,
                convert_to_cz='auto',
            )

    #we should get stim circuit from that and the we can just return it
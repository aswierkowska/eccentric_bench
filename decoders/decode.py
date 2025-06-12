import logging
import stim

from .bp_osd import bposd_chk, bposd_batch
from .mwpm import mwpm


def decode(
    code_name: str, circuit: stim.Circuit, num_shots: int, decoder: str
) -> float:
    if decoder == "mwpm":
        try:
            return mwpm(circuit, num_shots)
        except Exception as e:
            logging.error(f"MWPM failed: {e}")
    elif decoder == "bposd":
        try:
            if code_name == "steane" or code_name == "bacon":
                return bposd_chk(circuit, num_shots)
            else:
                return bposd_batch(circuit, num_shots)
        except Exception as e:
            logging.error(f"BP-OSD failed: {e}")
            return None
    else:
        raise NotImplementedError

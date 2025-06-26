import logging
import stim

from .bp_osd import bposd_chk, bposd_batch
from .mwpm import mwpm


def decode(
        code_name: str, circuit: stim.Circuit, num_shots: int, decoder: str, backend_name: str, error_type: str
) -> float:
    approximate_disjoint_errors = False
    if (backend_name.split("_")[0] == "variance" or backend_name.split("_")[0] == "real") and (error_type == "variance" or error_type.split("_")[0] == "real"):
        approximate_disjoint_errors = True

    if decoder == "mwpm":
        try:
            return mwpm(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"MWPM failed: {e}")
    elif decoder == "bposd":
        try:
            if code_name == "steane" or code_name == "bacon":
                return bposd_chk(circuit, num_shots, approximate_disjoint_errors)
            else:
                return bposd_batch(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"BP-OSD failed: {e}")
            return None
    else:
        raise NotImplementedError

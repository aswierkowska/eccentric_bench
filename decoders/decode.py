import stim
import pymatching
from stimbposd import (
    BPOSD,
)  # doesn't work with current ldpcv2 code  pip install -U ldpc==0.1.60
from ldpc import bposd_decoder


def decode(decoder: str, detector_error_model: stim.DetectorErrorModel, detection_events): # TODO: add detection_events type
    if decoder == "mwpm":
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    elif decoder == "bposd":
        # TODO: adjust BP-OSD
        matcher = BPOSD(detector_error_model, max_bp_iters=40)
    else:
        raise NotImplementedError
    return matcher.decode_batch(detection_events)
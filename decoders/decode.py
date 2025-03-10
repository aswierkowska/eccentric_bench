import stim
import pymatching
from stimbposd import (
    BPOSD,
)  # doesn't work with current ldpcv2 code  pip install -U ldpc==0.1.60
from ldpc import bposd_decoder
from .bp_osd import modified_bposd_decoder


def decode(decoder: str, dem: stim.DetectorErrorModel, detection_events): # TODO: add detection_events type
    if decoder == "mwpm":
        matcher = pymatching.Matching.from_detector_error_model(dem)
    elif decoder == "bposd":
        # TODO: adjust BP-OSD
        modified_bposd_decoder(dem, 12, 500)
        matcher = BPOSD(dem, max_bp_iters=40)
    else:
        raise NotImplementedError
    return matcher.decode_batch(detection_events)
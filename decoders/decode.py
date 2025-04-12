import subprocess

import stim
import pymatching
import numpy as np
from stimbposd import (
    BPOSD,
)  # doesn't work with current ldpcv2 code  pip install -U ldpc==0.1.60
from ldpc import bposd_decoder
from .bp_osd import modified_bposd_decoder

# TODO: for now returns logical error rate
# TODO: add detection_events type
def decode(code_name: str, circuit: stim.Circuit, num_shots: int, decoder: str) -> float:
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )
    
    dem = circuit.detector_error_model() # TODO do we need those: decompose_errors=True, approximate_disjoint_errors=True
    if decoder == "mwpm":
        matcher = pymatching.Matching.from_detector_error_model(dem)
        predictions = matcher.decode_batch(detection_events)
        num_errors = 0
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        return num_errors / num_shots
    elif decoder == "bposd":
        # TODO: adjust BP-OSD
        if code_name == "gross":
            return modified_bposd_decoder(dem, 12, 100)
        else:
            matcher = BPOSD(dem, max_bp_iters=1000, bp_method="minimum_sum_log", osd_method="osd_cs", osd_order=10)
            predictions = matcher.decode_batch(detection_events)
            num_errors = 0
            for shot in range(num_shots):
                actual_for_shot = observable_flips[shot]
                predicted_for_shot = predictions[shot]
                if not np.array_equal(actual_for_shot, predicted_for_shot):
                    num_errors += 1
            return num_errors / num_shots
    else:
        raise NotImplementedError
import pymatching
import stim
import numpy as np


def mwpm(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool):
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    predictions = matcher.decode_batch(detection_events)
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors / num_shots

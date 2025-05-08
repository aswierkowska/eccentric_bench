import stim
from typing import Tuple
from .noise import *

# From https://arxiv.org/pdf/2408.13687
willow_err_prob = {
    "P_CZ": 2.8e-3,
    "P_CZ_CROSSTALK": 5.5e-4,
    "P_CZ_LEAKAGE": 2.0e-4,
    "P_IDLE": 0.9e-2,
    "P_READOUT": 0.8e-2,
    "P_RESET": 1.5e-3,
    "P_SQ": 6.2e-4,
    "P_LEAKAGE": 2.5e-4
}

"""
# From https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf
willow_err_prob = {
    "P_CZ": 0.33e-2, # Two-qubit gate error (mean, simultaneous) for CZ [1]
    "P_CZ_CROSSTALK": None,
    "P_CZ_LEAKAGE": None,
    "P_IDLE": None,
    "P_READOUT": 0.77e-2,
    "P_RESET": None,
    "P_SQ": 0.035e-2, # Single-qubit gate error (mean, simultaneous) [1]
    "P_LEAKAGE": None
}
"""

# TODO: remove init, modify get noise, pass crosstalk as crosstalk and figure out leakage
class WillowNoise(NoiseModel):
    def __init__(self, qt, idle, measure_reset_idle, noisy_gates, noisy_gates_connection=None, use_correlated_parity_measurement_errors=False):
        super().__init__(
            idle=idle,
            measure_reset_idle=measure_reset_idle,
            noisy_gates=noisy_gates,
            use_correlated_parity_measurement_errors=use_correlated_parity_measurement_errors,
        )
        self.qt = qt


    @staticmethod
    def get_noise(qt) -> 'WillowNoise':
        return WillowNoise(
            qt=qt,
            idle=willow_err_prob["P_IDLE"],
            measure_reset_idle=willow_err_prob["P_RESET"],
            noisy_gates={
                "CX": willow_err_prob["P_CZ"],
                "CZ": willow_err_prob["P_CZ"],
                "CZ_CROSSTALK": willow_err_prob["P_CZ_CROSSTALK"],
                "CZ_LEAKAGE": willow_err_prob["P_CZ_LEAKAGE"],
                "R": willow_err_prob["P_RESET"],
                "H": willow_err_prob["P_SQ"],
                "M": willow_err_prob["P_READOUT"],
                "MPP": willow_err_prob["P_READOUT"],
            },
            use_correlated_parity_measurement_errors=True
        )
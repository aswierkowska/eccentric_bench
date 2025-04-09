import dataclasses
from .noise import NoiseModel

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

@dataclasses.dataclass(frozen=True)
class WillowNoise(NoiseModel):
    
    @staticmethod
    def get_noise() -> 'NoiseModel':
        return NoiseModel(
            idle=willow_err_prob["P_IDLE"],
            measure_reset_idle=willow_err_prob["P_RESET"],
            noisy_gates={
                # TODO: should not be CX
                "CX": willow_err_prob["P_CZ"],
                "CZ": willow_err_prob["P_CZ"],
                "CZ_CROSSTALK": willow_err_prob["P_CZ_CROSSTALK"],
                "CZ_LEAKAGE": willow_err_prob["P_CZ_LEAKAGE"],
                "R": willow_err_prob["P_RESET"],
                # TODO: should not be H
                "H": willow_err_prob["P_SQ"],
                "M": willow_err_prob["P_READOUT"],
                "MPP": willow_err_prob["P_READOUT"],
            },
            use_correlated_parity_measurement_errors=True
    )
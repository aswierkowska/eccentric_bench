from .noise import *

class WillowNoise(NoiseModel):
    
    @staticmethod
    def get_noise(qt) -> 'NoiseModel':
        # From https://arxiv.org/pdf/2408.13687
        return NoiseModel(
            sq=6.2e-4,
            tq=2.8e-3,
            idle=0.9e-2,
            crosstalk=5.5e-4,
            leakage=2.5e-4,
            leakage_propagation=2.0e-4,
            reset=1.5e-3,
            measure=0.8e-2, # In the paper: readout
            qt=qt
        )

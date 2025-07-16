from .noise import *
from backends import QubitTracking

# TODO: Should the crosstalk be more granular?
# crosstalk_gates={
#    "R": 0.04567e-4,
#    "SQ": 0.066e-5,
#    "TQ": 0.066e-5,
#    "M": 0.03867e-4,
#}

class ApolloNoise(NoiseModel):
    # H2 rescaled according to roadmap: https://www.quantinuum.com/press-releases/quantinuum-unveils-accelerated-roadmap-to-achieve-universal-fault-tolerant-quantum-computing-by-2030
    @staticmethod
    def get_noise(qt: QubitTracking) -> 'NoiseModel':
        return NoiseModel(
            sq=8.0e-5 / 10,
            tq=1.4e-3 / 10,
            idle=5.3e-4 / 10,
            crosstalk=6.3e-6 / 10,
            measure=1.33e-3 / 10,
            leakage=4.3e-4 / 10,
            remote=6.3e-06 / 10,
            gate_times={
                # As suggested by Quantinuum FAQ: https://arxiv.org/pdf/2003.01293
                "SQ": 5 * 1e-6,
                "TQ": 25 * 1e-6,
                "M": 60 * 1e-6,
                "R": 10 * 1e-6,
                "REMOTE": 0,
            },
            qt=qt
        )
    
    # H2:    
    #def get_noise(qt: QubitTracking) -> 'NoiseModel':
    #    # Values from https://github.com/CQCL/quantinuum-hardware-specifications/blob/main/qtm_spec/combined_analysis.py
    #    return NoiseModel(
    #        sq=8.0e-5,
    #        tq=1.4e-3,
    #        idle=5.3e-4,
    #        crosstalk=6.3e-6,
    #        measure=1.33e-3,
    #        leakage=4.3e-4,
    #        remote=6.3e-06,
    #        gate_times={
    #            "REMOTE": 0,
    #        },
    #        qt=qt
    #    )

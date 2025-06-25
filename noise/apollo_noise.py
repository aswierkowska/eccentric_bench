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

    @staticmethod
    def get_noise(qt: QubitTracking) -> 'NoiseModel':
        # TODO: get more detailed values from https://github.com/CQCL/quantinuum-hardware-specifications/blob/main/qtm_spec/combined_analysis.py or here: https://arxiv.org/pdf/2406.02501
        # Values taken from Quantinuum H2 and rescaled according to their roadmap
        return NoiseModel(
            sq=0.000002,
            tq=0.0001,
            idle=0.00223,
            crosstalk=0.66e-6,
            measure=0.0001,
            remote=2.23e-4, # TODO SCALE
            gate_times={
                "REMOTE": 4, # TODO ADD
            },
            qt=qt
            # backend seems unnecessary as t1 is a few minutes
        )
from .noise import *
from backends import QubitTracking, FakeInfleqtionBackend

class InfleqtionNoise(NoiseModel):

    @staticmethod
    def get_noise(qt: QubitTracking, backend: FakeInfleqtionBackend) -> 'NoiseModel':
        return NoiseModel(
            # https://arxiv.org/pdf/2408.08288
            sq=0.000971, # Derived from local RZ gate fidelity of 99.902(8)%
            tq=0.00646, # Derived from CZ fidelity of 99.35(4)% 
            crosstalk=0, # included in gate error
            leakage=0, # included in gate error
            measure=0.0224, # SPAM, no additional reset error
            gate_times={
                "SQ": 2.5e-7,
                "TQ": 4.16e-7,
                "M": 6.0e-3,
                "R": 1.22e-2,
                "REMOTE": 2.182e-4,
            },
            qt=qt,
            backend=backend
        )
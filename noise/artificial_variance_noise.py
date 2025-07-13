from .noise import *
from backends import QubitTracking


class ArtificialVarianceNoise(NoiseModel):
    @staticmethod
    def get_noise(
        p: float,
        qt: QubitTracking,
        backend
    ) -> 'NoiseModel':
        return NoiseModel(
            sq=p,
            tq=p,
            measure=p,
            reset=p,
            gate_times={
                "SQ": 50 * 1e-9,
                "TQ": 70 * 1e-9,
                "M": 70 * 1e-9,
                "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend
        )

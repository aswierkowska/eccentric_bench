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
            idle=p,
            measure=p,
            reset=p,
            qt=qt,
            backend=backend
        )
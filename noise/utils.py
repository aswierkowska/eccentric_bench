from .willow_noise import WillowNoise
from .flamingo_noise import FlamingoNoise
from .apollo_noise import ApolloNoise
from .infleqtion_noise import InfleqtionNoise
from .artificial_noise import ArtificialNoise
from .artificial_variance_noise import ArtificialVarianceNoise
from .heron_noise import HeronNoise
from backends import *
from typing import Union

def get_noise_model(error_type: str, qt: QubitTracking, p: float = None, backend: Union[FakeIBMFlamingo, FakeInfleqtionBackend] = None):
    if p:
        if error_type == "sd6":
            return ArtificialNoise.SD6(p, qt)
        elif error_type == "pm3":
            return ArtificialNoise.PC3(p, qt)
        elif error_type == "em3_1":
            return ArtificialNoise.EM3_v1(p, qt)
        elif error_type == "em3_2":
            return ArtificialNoise.EM3_v2(p, qt)
        elif error_type == "si1000":
            return ArtificialNoise.SI1000(p, qt)
        elif error_type == "modsi1000":
            return ArtificialNoise.modSI1000(p, qt)
        elif error_type == "constant":
            return ArtificialNoise.constant(p, qt)
        elif error_type == "variance":
            return ArtificialVarianceNoise.get_noise(p, qt, backend)
    if error_type == "real_willow":
        return WillowNoise.get_noise(qt)
    elif error_type == "real_flamingo" and backend:
        return FlamingoNoise.get_noise(qt, backend)
    elif error_type == "real_infleqtion" and backend:
        return InfleqtionNoise.get_noise(qt, backend)
    elif error_type == "real_apollo":
        return ApolloNoise.get_noise(qt)
    elif error_type == "real_heron":
        return HeronNoise.get_noise(qt, backend)
    raise NotImplementedError

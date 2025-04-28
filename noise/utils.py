from .noise import NoiseModel
from .willow_noise import WillowNoise
from .flamingo_noise import FlamingoNoise
from .apollo_noise import ApolloNoise
from .infleqtion_noise import InfleqtionNoise
from backends import *

def get_noise_model(error_type: str, p: float = None, qt: QubitTracking = None, backend: FakeIBMFlamingo = None):
    if p:
        if error_type == "sd6":
            return NoiseModel.SD6(p)
        elif error_type == "pm3":
            return NoiseModel.PC3(p)
        elif error_type == "em3_1":
            return NoiseModel.EM3_v1(p)
        elif error_type == "em3_2":
            return NoiseModel.EM3_v2(p)
        elif error_type == "si1000":
            return NoiseModel.SI1000(p)
    # TODO: add qt everywhere for
    if qt:
        if error_type == "willow":
            return WillowNoise.get_noise(qt)
        elif error_type == "flamingo" and backend:
            return FlamingoNoise.get_noise(qt, backend)
        elif error_type == "infleqtion":
            return InfleqtionNoise.get_noise(qt)
        #elif error_type == "aquila":
        #    return AquilaNoise.get_noise(qt)
        elif error_type == "apollo":
            return ApolloNoise.get_noise(qt)
    raise NotImplementedError

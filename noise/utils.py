from .noise import NoiseModel
from .willow_noise import WillowNoise
from .flamingo_noise import FlamingoNoise
from .aquila_noise import AquilaNoise
from .apollo_noise import ApolloNoise
from backends import QubitTracking

def get_noise_model(name: str, p: float = None, qt: QubitTracking = None):
    if p:
        if name == "sd6":
            return NoiseModel.SD6(p)
        elif name == "pm3":
            return NoiseModel.PC3(p)
        elif name == "em3_1":
            return NoiseModel.EM3_v1(p)
        elif name == "em3_2":
            return NoiseModel.EM3_v2(p)
        elif name == "si1000":
            return NoiseModel.SI1000(p)
    if qt:
        if name == "willow":
            return WillowNoise.get_noise()
        elif name == "flamingo":
            return FlamingoNoise.get_noise()
        elif name == "aquila":
            return AquilaNoise.get_noise()
        elif name == "apollo":
            return ApolloNoise.get_noise()
    raise NotImplementedError
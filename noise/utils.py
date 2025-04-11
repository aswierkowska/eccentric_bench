from .noise import NoiseModel
from .willow_noise import WillowNoise

def get_noise_model(name, p):
    #return NoiseModel.SD6(p)
    return WillowNoise.get_noise()
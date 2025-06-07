from .noise import *
from backends import QubitTracking, FakeInfleqtionBackend

class InfleqtionNoise(NoiseModel):

    @staticmethod
    def get_noise(qt: QubitTracking, backend: FakeInfleqtionBackend) -> 'NoiseModel':
        return NoiseModel(
            # TODO: fact-check and cite paper here
            sq=0.00098, # Derived from local RZ gate fidelity of 99.902(8)%
            tq=0.0065, # Derived from CZ fidelity of 99.35(4)% 
            crosstalk=0.0001, # Based on crosstalk estimation smaller than 10^-4 
            leakage=0.0010, # State-averaged atom loss probability of 0.9(3)% during NDSSR 
            measure=0.004, # Derived from bright-dark discrimination fidelity of 99.6(2)% 
            # TODO: shuttle if we account for heating
            gate_times={
                "SQ": 2.5e-7,
                "TQ": 4.16e-7,
                "M": 6.0e-3,
                "R": 4.1e-6,
                "REMOTE": 0, # TODO ADD
            },
            qt=qt,
            backend=backend
        )
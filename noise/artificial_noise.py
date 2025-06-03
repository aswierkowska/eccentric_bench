from .noise import *

class ArtificialNoise(NoiseModel):

    @staticmethod
    def SD6(p: float, qt) -> 'NoiseModel':
        return NoiseModel(
            sq=p,
            idle=p,
            measure=0,
            reset=0,
            qt=qt,
            noisy_gates={
                "CX": p,
                "R": p,
                "M": p,
            },
        )

    @staticmethod
    def PC3(p: float, qt) -> 'NoiseModel':
        return NoiseModel(
            sq=p,
            tq=p,
            idle=p,
            measure=0,
            reset=0,
            qt=qt,
            noisy_gates={
                "R": p,
                "M": p,
            },
        )

    @staticmethod
    def EM3_v1(p: float, qt) -> 'NoiseModel':
        """EM3 but with measurement flip errors independent of measurement target depolarization error."""
        return NoiseModel(
            idle=p,
            measure=0,
            reset=0,
            sq=p,
            qt=qt,
            noisy_gates={
                "R": p,
                "M": p,
                "MPP": p,
            },
        )

    @staticmethod
    def EM3_v2(p: float, qt) -> 'NoiseModel':
        """EM3 with measurement flip errors correlated with measurement target depolarization error."""
        return NoiseModel(
            sq=0,
            tq=0,
            idle=p,
            measure=0,
            reset=0,
            qt=qt,
            use_correlated_parity_measurement_errors=True,
            noisy_gates={
                "R": p/2,
                "M": p/2,
                "MPP": p,
            },
        )

    @staticmethod
    def SI1000(p: float, qt) -> 'NoiseModel':
        """Inspired by superconducting device."""
        return NoiseModel(
            sq=p / 10,
            idle=p / 10,
            measure=2 * p,
            reset=2 * p,
            qt=qt,
            noisy_gates={
                "CZ": p,
                "R": 2 * p,
                "M": 5 * p,
            },
        )
    
    @staticmethod
    def constant(p: float, qt) -> 'NoiseModel':
        """Inspired by superconducting device."""
        return NoiseModel(
            sq=p,
            tq=p,
            idle=p,
            measure=p,
            reset=p,
            qt=qt,
        )


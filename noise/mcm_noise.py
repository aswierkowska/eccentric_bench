from .noise import *
from backends import GridMCMBackend, QubitTracking


class MCMNoise(NoiseModel):

    @staticmethod
    def get_noise(
        qt: QubitTracking,
        backend,
        m_error_multiplier = 1,
        m_time_multiplier = 1,
        decoding_time = 0
    ) -> 'NoiseModel':
        m_error_multiplier = float(m_error_multiplier)
        m_time_multiplier = float(m_time_multiplier)
        decoding_time = float(decoding_time)
        p = 1e-4
        return NoiseModel(
            sq=p,
            tq=5 * p,
            measure=5 * p * m_error_multiplier,
            gate_times={
                "SQ": 50 * 1e-9,
                "TQ": 70 * 1e-9,
                "M": 1000 * 1e-9 * m_time_multiplier + decoding_time * 1e-6,
                "REMOTE": (300 * 1e-9) / (2.2222222222222221e-10 * 1e9) * (2.2222222222222221e-10 * 1e9),
                "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend
        )

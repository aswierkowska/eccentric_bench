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
        # IBM FEZ
        return NoiseModel(
            sq=2.843 * 1e-4 / 10,
            tq=2.709*1e-3 / 10,
            measure=1.46*1e-2 * m_error_multiplier / 10,
            gate_times={
                "SQ": 24 * 1e-9,
                "TQ": 68 * 1e-9,
                "M": 1000 * 1e-9 * m_time_multiplier + decoding_time * 1e-6,
                "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend
        )

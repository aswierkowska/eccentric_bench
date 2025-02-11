class BasicBackend(GenericBackendV2):
    def __init__(
        self,
        num_qubits: int,
        basis_gates: list[str] | None = None,
        *,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        control_flow: bool = False,
        calibrate_instructions: bool | InstructionScheduleMap | None = None,
        noise_settings: dict,
        dtm: float | None = None,
        seed: int | None = None,
    ):

        self.noise_settings = noise_settings
        super().__init__(num_qubits, basis_gates,
            coupling_map=coupling_map,
            control_flow=control_flow,
            calibrate_instructions=calibrate_instructions,
            dtm=dtm,
            seed=seed)

    def _get_noise_defaults(self, name: str, num_qubits: int) -> tuple:
        if name in self.noise_settings:
            return self.noise_settings[name]

        if num_qubits == 1:
            return _NOISE_DEFAULTS_FALLBACK["1-q"]
        return _NOISE_DEFAULTS_FALLBACK["multi-q"]
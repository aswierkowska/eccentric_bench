import sys
import os

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))
import logging
import stim
from metrics import get_
from qiskit_qec.utils import get_stim_circuits
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging




if __name__ == "__main__":
        experiment_name = "Code_statistics"
        setup_experiment_logging(experiment_name)

        for code_name in ["steane", "surface", "color", "gross", "hh", "bacon"]:
            code = get_code(code_name, d, cycles)

            detectors, logicals = code.stim_detectors()
            for state, qc in code.circuit.items():
                stim_circuit = get_stim_circuits(
                    code.circuit[state], detectors=detectors, logicals=logicals
                )[0][0]
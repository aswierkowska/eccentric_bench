import logging
import os
import csv
import json
import datetime

def create_experiment_directory(experiment_name):
    directory = f"experiment_results/{experiment_name}"
    os.makedirs(directory, exist_ok=True)
    return directory

def save_results_to_csv(data, experiment_name):
    try:
        directory = create_experiment_directory(experiment_name)
        filename = os.path.join(directory, "results.csv")

        file_exists = os.path.isfile(filename)
        with open(filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    except Exception as e:
        logging.error(f"ERROR SAVING: {data}")

def save_experiment_metadata(experiment, experiment_name):
    directory = create_experiment_directory(experiment_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"metadata_{timestamp}.json")

    with open(filename, "w") as f:
        json.dump(experiment, f, indent=4)

def setup_experiment_logging(experiment_name):
    directory = create_experiment_directory(experiment_name)
    log_filename = os.path.join(directory, "experiment.log")

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    for logger_name in [
        "qiskit.transpiler",
        "qiskit.transpiler.passes",
        "qiskit.transpiler.passmanager",
        "qiskit",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger(logger_name).propagate = False

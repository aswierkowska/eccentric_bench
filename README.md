# eccentric_bench

Eccentric Bench is a project focused on realistic benchmarking Quantum Error Correction Codes.

## Installation (Linux)

Follow these steps to set up the project locally:

1. **Clone the repository:**

    ```bash
    git clone git@github.com:aswierkowska/eccentric_bench.git
    cd eccentric_bench
    ```

2. **Install `virtualenv`:**

    ```bash
    pip install virtualenv
    ```

3. **Create a virtual environment:**

    ```bash
    virtualenv venv
    ```

4. **Activate the virtual environment:**

    ```bash
    source venv/bin/activate
    ```

5. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

6. **Initialize the submodule:**

    ```bash
    git submodule update --init --recursive
    ```

7. **IBM API token:**

    This project requires using an IBM API token, which should be saved in your working environment. Please follow the instructions from the IBM guide to set up the token:

    [IBM Quantum API Setup Guide](https://docs.quantum.ibm.com/guides/setup-channel)

## Running the Project

Once the environment is set up, you can run the project with:

```bash
python3 main.py
```

The results will be saved in the `qecc_benchmark.log`.

## Possible Erros

If you encounter issues related to building C extension files in the qiskit-qec, run the following commands to build the necessary C files from source:

```bash
cd external/qiskit_qec
python setup.py build_ext --inplace
```
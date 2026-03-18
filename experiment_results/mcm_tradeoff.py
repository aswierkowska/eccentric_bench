"""
evaluate_mcm_error_latency_pairs.py
==================================
Evaluates how specific mid-circuit measurement (MCM) error rates and
latencies affect the logical error rate (LER) of a surface-code lattice-
surgery logical measurement (merge=True).

Fixed parameters
----------------
- CNOT latency        : CNOT_LATENCY_NS
- CNOT error rate     : CNOT_ERROR
- T1, T2              : fixed per hardware preset
- idle_multiplier     : 1 (Google) or 3 (IBM)
- No synchronization  : sync=None

Swept parameters
----------------
- Distance            : DISTANCES
- Latency + Error     : custom pairs [(latency_ns, measure_error), ...]
"""

import sys
import os
import csv
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

# Code distances (odd integers only)
DISTANCES = [7, 9, 11]

# List of (latency_ns, measure_error) pairs
LATENCY_ERROR_PAIRS = [
    (250, 0.05),
    (500, 0.0292),
    (750, 0.0104),
    (1000, 0.0081),
]

# Fixed CNOT parameters
CNOT_LATENCY_NS = 70
CNOT_ERROR      = 0.0002

# Hardware decoherence
T1_US    = 130
T2_US    = 170
IDLE_MUL = 3

# Lattice surgery basis
BASIS    = 'Z'
LS_BASIS = 'X'

# Shots per configuration
NUM_SHOTS = 100_000

# Number of parallel worker processes (None → all CPUs)
NUM_PROCS = None

# Output CSV path
OUTPUT_CSV = 'mcm_tradeoff_herqules_futuristic.csv'

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _run_one(args):
    """
    Simulate one (distance, latency, measure_error) configuration.
    """
    d, latency_ns, measure_error = args

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
    from circuit_4 import circuit  # noqa

    sim = circuit(
        distance          = d,
        num_patches_x     = 20,
        num_patches_y     = 20,
        spacing           = 1,
        disable_noise     = False,
        # --- fixed MCM parameters ---
        fixed_measure_latency = latency_ns,
        fixed_measure_noise   = measure_error,
        # --- fixed CNOT parameters ---
        fixed_cnot_latency    = CNOT_LATENCY_NS,
        fixed_cnot_noise      = CNOT_ERROR,
        # --- decoherence ---
        fixed_t1         = T1_US,
        fixed_t2         = T2_US,
        idle_multiplier  = IDLE_MUL,
        # --- logical operator ---
        merge    = True,
        basis    = BASIS,
        ls_basis = LS_BASIS,
        sync     = None,
        rounds_per_op = d + 1,
    ).from_string('qreg q[2];')

    ler, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=NUM_SHOTS)
    return (d, latency_ns, measure_error, ler)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # Build full list of configs: each distance × each latency/error pair
    configs = [(d, lat, err) for d in DISTANCES for lat, err in LATENCY_ERROR_PAIRS]
    print(f"Evaluating custom MCM latency/error pairs")
    print(f"  Distances         : {DISTANCES}")
    print(f"  Latency/Error Pairs: {LATENCY_ERROR_PAIRS}")
    print(f"  Total configs     : {len(configs)}")
    print(f"  Shots per config  : {NUM_SHOTS:,}")
    print(f"  Output file       : {OUTPUT_CSV}")
    print()

    num_procs = NUM_PROCS or mp.cpu_count()

    results = []
    with mp.Pool(num_procs) as pool:
        for result in tqdm(pool.imap_unordered(_run_one, configs),
                           total=len(configs),
                           desc='Simulating'):
            results.append(result)

    results.sort(key=lambda x: (x[0], x[1]))  # sort by distance, then latency

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['distance', 'measure_latency_ns', 'measure_error', 'logical_error_rate'])
        writer.writerows(results)

    # Pretty-print table
    print()
    print(f"{'distance':>10}  {'latency(ns)':>12}  {'measure_error':>15}  {'LER':>12}")
    print('-' * 60)
    for d, lat, err, ler in results:
        print(f"{d:>10}  {lat:>12}  {err:>15.4f}  {ler:>12.6f}")

    print()
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()

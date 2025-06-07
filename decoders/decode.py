import subprocess
import logging
import stim
import pymatching
import numpy as np
import chromobius
from stimbposd import (
    BPOSD,
)  # doesn't work with current ldpcv2 code  pip install -U ldpc==0.1.60
from ldpc import bposd_decoder
from .bp_osd import modified_bposd_decoder
from scipy.sparse import csc_matrix
from typing import Dict, List, FrozenSet

def dict_to_csc_matrix(elements_dict, shape):
    # Constructs a `scipy.sparse.csc_matrix` check matrix from a dictionary `elements_dict`
    # giving the indices of nonzero rows in each column.
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


def dem_to_check_matrices(dem: stim.DetectorErrorModel, return_col_dict=False):

    DL_ids: Dict[str, int] = {}  # detectors + logical operators
    L_map: Dict[int, FrozenSet[int]] = {}  # logical operators
    priors_dict: Dict[int, float] = {}  # for each fault

    def handle_error(prob: float, detectors: List[int], observables: List[int]) -> None:
        dets = frozenset(detectors)
        obs = frozenset(observables)
        key = " ".join([f"D{s}" for s in sorted(dets)] + [f"L{s}" for s in sorted(obs)])

        if key not in DL_ids:
            DL_ids[key] = len(DL_ids)
            priors_dict[DL_ids[key]] = 0.0

        hid = DL_ids[key]
        L_map[hid] = obs
        #         priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])
        priors_dict[hid] += prob

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[int] = []
            frames: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    frames.append(t.val)
            handle_error(p, dets, frames)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    check_matrix = dict_to_csc_matrix(
        {
            v: [int(s[1:]) for s in k.split(" ") if s.startswith("D")]
            for k, v in DL_ids.items()
        },
        shape=(dem.num_detectors, len(DL_ids)),
    )
    observables_matrix = dict_to_csc_matrix(
        L_map, shape=(dem.num_observables, len(DL_ids))
    )
    priors = np.zeros(len(DL_ids))
    for i, p in priors_dict.items():
        priors[i] = p

    if return_col_dict:
        return check_matrix, observables_matrix, priors, DL_ids
    return check_matrix, observables_matrix, priors

# TODO: for now returns logical error rate
# TODO: add detection_events type
def decode(code_name: str, circuit: stim.Circuit, num_shots: int, decoder: str) -> float:
    if decoder == "mwpm":
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            num_shots, separate_observables=True
        )
        dem = circuit.detector_error_model()
        matcher = pymatching.Matching.from_detector_error_model(dem)
        predictions = matcher.decode_batch(detection_events)
        num_errors = 0
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        return num_errors / num_shots
    elif decoder == "chromobius":
        dets, actual_obs_flips = circuit.compile_detector_sampler().sample(
            shots=num_shots,
            separate_observables=True,
            bit_packed=True,
        )
        decoder = chromobius.compile_decoder_for_dem(circuit.detector_error_model())
        predicted_obs_flips = decoder.predict_obs_flips_from_dets_bit_packed(dets)
        return np.count_nonzero(np.any(predicted_obs_flips != actual_obs_flips, axis=1))
    elif decoder == "bposd":
        # TODO: adjust BP-OSD
        try:
            if code_name != "":
                dem = circuit.detector_error_model()
                chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)
                num_row, num_col = chk.shape
                chk_row_wt = np.sum(chk, axis=1)
                chk_col_wt = np.sum(chk, axis=0)
                print(f"check matrix shape {chk.shape}, max (row, column) weight ({np.max(chk_row_wt)}, {np.max(chk_col_wt)}),",
                    f"min (row, column) weight ({np.min(chk_row_wt)}, {np.min(chk_col_wt)})")
                #bpd = BpOsdDecoder(
                #        chk, # the parity check matrix
                #        channel_probs=list(priors), # assign error_rate to each VN
                #        max_iter=10000, # the maximum number of iterations for BP
                #        bp_method="minimum_sum", # messages are not clipped, may have numerical issues
                #        ms_scaling_factor=1.0, # min sum scaling factor. If set to zero, the variable scaling factor method is used
                #        osd_method="OSD_CS", # the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                #        osd_order=10, # the osd search depth, not specified in v1, later specified to be 7 in their arXiv v2
                #)

                # New decoder
                
                bpd = bposd_decoder(
                    chk,  # the parity check matrix
                    channel_probs=list(priors),  # assign error_rate to each qubit. This will override "error_rate" input variable
                    max_iter=10000,  # the maximum number of iterations for BP
                    bp_method="minimum_sum",  # messages are not clipped, may have numerical issues
                    ms_scaling_factor=1.0,  # min sum scaling factor. If set to zero the variable scaling factor method is used
                    osd_method="osd_cs",  # the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                    osd_order=10,  # the osd search depth, not specified in [1]
                    input_vector_type="syndrome",  # "received_vector"
                )

                """
                # Decoder before May
                bpd = bposd_decoder(
                    chk, # the parity check matrix
                    error_rate=0, # does not matter as channel_prob is set
                    channel_probs=priors, # assign error_rate to each qubit. This will override "error_rate" input variable
                    max_iter=10000, # the maximum number of iterations for BP
                    bp_method="minimum_sum_log", # messages are not clipped, may have numerical issues
                    ms_scaling_factor=1.0, # min sum scaling factor. If set to zero the variable scaling factor method is used
                    osd_method="osd_cs", # the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                    osd_order=10, # the osd search depth, not specified in [1]
                    input_vector_type="syndrome", # "received_vector"
                )
                """
                dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
                det_data, obs_data, err_data = dem_sampler.sample(shots=num_shots, return_errors=False, bit_packed=False)
                print("detector data shape", det_data.shape)
                print("observable data shape", obs_data.shape)

                num_err = 0
                num_flag_err = 0
                for i in range(num_shots):
                    e_hat = bpd.decode(det_data[i])
                    num_flag_err += ((chk @ e_hat + det_data[i]) % 2).any()
                    ans = (obs @ e_hat + obs_data[i]) % 2
                    num_err += ans.any()
                print(f"Flagged Errors: {num_flag_err}/{num_shots}") # expect 0 for OSD
                print(f"Logical Errors: {num_err}/{num_shots}")
                p_l = num_err / num_shots
                return p_l
                #return modified_bposd_decoder(dem, 12, 100)
            else:
                matcher = BPOSD(dem, max_bp_iters=1000, bp_method="minimum_sum_log", osd_method="osd_cs", osd_order=10)
                predictions = matcher.decode_batch(detection_events)
                num_errors = 0
                for shot in range(num_shots):
                    actual_for_shot = observable_flips[shot]
                    predicted_for_shot = predictions[shot]
                    if not np.array_equal(actual_for_shot, predicted_for_shot):
                        num_errors += 1
                return num_errors / num_shots
        except Exception as e:
            logging.error("BP-OSD failed")
            return None
    else:
        raise NotImplementedError

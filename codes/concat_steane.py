#Please forgive me for this code

from abc import abstractmethod, abstractproperty
import sys
import cirq
import stim 
import stimcirq
import dataclasses
from typing import Dict,Iterable, List, Union,TextIO
from enum import Enum

from qiskit_qec.circuits import StimCodeCircuit

class MeasureReset(stimcirq.MeasureAndOrResetGate):
    def __init__(self) -> None:
        super().__init__(
            measure=True, reset=False, basis="Z", invert_measure=False, key="m"
        )

MR = MeasureReset()

def filter_ticks(circuit: stim.Circuit) -> stim.Circuit:
    """Filters ticks out of a stim.Circuit."""
    result = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            result += handle_repeat_block(op)
        elif op.name == "TICK":
            continue
        # elif op.name == "R":
        #     continue
        # elif op.name == "M":
        #     result += stim.Circuit(
        #         str(stim.CircuitInstruction("MR", op.targets_copy()))
        #     )
        else:
            result += stim.Circuit(str(op))

    return result


def handle_repeat_block(block) -> stim.Circuit:
    return filter_ticks(block.body_copy()) * block.repeat_count

class JobType(Enum):
    ONLINE = 1
    OFFLINE = 2

    def from_arg(job_type):
        if job_type == "online":
            return JobType.ONLINE
        elif job_type == "offline":
            return JobType.OFFLINE
        else:
            raise ValueError(f"Unrecognized job type {job_type}.")


class AncillaType(Enum):
    SINGLE_FLAG = 1
    BARE = 2

    def from_arg(anc):
        if anc == "single_flag":
            return AncillaType.SINGLE_FLAG
        elif anc == "bare":
            return AncillaType.BARE
        else:
            raise ValueError(f"Unrecognized ancilla type {anc}.")


@dataclasses.dataclass
class Params:
    job_type: JobType
    ancilla_type: AncillaType
    p: float
    separate_measure_reset: bool = False
    debug: bool = dataclasses.field(default=False, repr=False)
    stdout: TextIO = dataclasses.field(default=sys.stdout, repr=False)

    def num_rounds_for_offline(self, t):
        return 1

@dataclasses.dataclass
class OrderedObservable:
    pauli: str
    ordering: List[int]


OBSERVABLE_LIKE = Union[str, cirq.PauliString, OrderedObservable]

class OfflineMemoryCircuit:
    def __init__(
        self,
        stabilizer_generators: List[str],
        logical_observables: List[str],
        qid_map: Dict[cirq.Qid, int],
        n_rounds: int,
    ):
        # super().__init__(stabilizer_generators + logical_observables, qid_map)
        self.n_rounds = n_rounds
        self.stabilizer_generators = stabilizer_generators
        self.logical_observables = logical_observables
        self.qid_map = qid_map

        self.n_gens = len(stabilizer_generators)
        self.n_logicals = len(logical_observables)

    @abstractmethod
    def reset_all(self):
        pass

    @abstractproperty
    def n_meas(self):
        pass

    def gauge_detectors(self) -> stim.Circuit:
        """These are for the first round."""
        ops = []
        for i in range(self.n_meas):
            ops.append(f"DETECTOR rec[{i - self.n_meas}]")
        return stim.Circuit("\n".join(ops))

    @abstractmethod
    def measure_round(self) -> stim.Circuit:
        pass

    def generate(self):
        #TODO what is the purpose of all the rounds
        circuit = stim.Circuit()
        circuit += self.reset_all()
        circuit += self.measure_round()
        circuit += self.gauge_detectors()

        circuit += self.measure_round()
        circuit += self.round_detectors()

        round = stim.Circuit()
        round += self.measure_round()
        round += self.round_detectors(2)
        round += self.measure_round()
        round += self.round_detectors(1)

        circuit += round * self.n_rounds
        #with open("our_circuit.stim", "w") as f:
        #    f.write(str(circuit))
        return circuit
    
class ObservableMeasurement:
    def __init__(self, separate_measure_reset: bool = False) -> None:
        self._separate_measure_reset = separate_measure_reset

    @abstractmethod
    def measure(
        self,
        obs: OBSERVABLE_LIKE,
        qs: Iterable[cirq.Qid],
    ) -> List[cirq.Operation]:
        pass

    def measure_reset(self, anc: cirq.Qid) -> Iterable[cirq.Operation]:
        return (
            [MR(anc)]
            if not self._separate_measure_reset
            else [cirq.measure(anc), cirq.reset(anc)]
        )

class CSSObservableMeasurement(ObservableMeasurement):
    def _observable_qubits_and_type(self, obs: OBSERVABLE_LIKE, qs: Iterable[cirq.Qid]):
        pauli_qs = []
        if isinstance(obs, str):
            ops = set(c for c in obs)
            if (
                (ops != {"I", "X"})
                and (ops != {"I", "Z"})
                and (ops != {"I"})
                and (ops != {"Z"})
                and (ops != {"X"})
            ):
                raise ValueError(
                    f"only X and Z observables are supported, mixed observables aren't allowed, like {obs}"
                )
            xtype = "X" in obs
            for op, q in zip(obs, qs):
                if op != "I":
                    pauli_qs.append(q)
        elif isinstance(obs, OrderedObservable):
            xtype = obs.pauli == "X"
            pauli_qs = [qs[q] for q in obs.ordering]
        return xtype, pauli_qs

    def measure(
        self, obs: OBSERVABLE_LIKE, qs: Iterable[cirq.Qid]
    ) -> List[cirq.Operation]:
        res = []

        if not isinstance(obs, str) and not isinstance(obs, OrderedObservable):
            raise ValueError(
                "only str and OrderedObservable observables are supported currently."
            )

        xtype, pauli_qs = self._observable_qubits_and_type(obs, qs)

        print("HELP: ",xtype, pauli_qs)

        anc = qs[-1]
        qs = qs[:-1]
        if self._separate_measure_reset:
            res.append(cirq.reset(anc))

        if xtype:
            res.append(cirq.H(anc))

        cop = lambda q: cirq.CX(anc, q) if xtype else cirq.CX(q, anc)

        for q in pauli_qs:
            res.append(cop(q))
        if xtype:
            res.append(cirq.H(anc))
        if self._separate_measure_reset:
            res.append(cirq.measure(anc))
        else:
            res += self.measure_reset(anc)
        return res


class BareAncillaOffline(OfflineMemoryCircuit):
    def __init__(
        self,
        stabilizer_generators: List[str],
        logical_observables: List[str],
        data: Iterable[cirq.Qid],
        syn: Iterable[cirq.Qid],
        n_rounds: int,
    ):
        q2i = {q: i for (i, q) in enumerate(data)}
        q2i.update({q: i + len(data) for (i, q) in enumerate(syn)})
        super().__init__(
            stabilizer_generators,
            logical_observables,
            q2i,
            n_rounds,
        )
        self.data = data
        self.syn = syn
        self.obs_measurer = CSSObservableMeasurement()
        self.measured_observables = (
            self.stabilizer_generators + self.logical_observables
        )

        print(self.measured_observables)

    @property
    def n_meas(self):
        return self.n_gens + self.n_logicals

    def _extractor(self, i) -> cirq.Circuit:
        return cirq.Circuit(
            self.obs_measurer.measure(
                self.measured_observables[i], self.observable_qubits(i)
            ),
            strategy=cirq.InsertStrategy.NEW,
        )

    def measure_round(self) -> stim.Circuit:
        result = stim.Circuit()


        for i in range(len(self.measured_observables)):
            result += stimcirq.cirq_circuit_to_stim_circuit(self._extractor(i),qubit_to_index_dict=self.qid_map)

        return result

    def observable_qubits(self, i) -> Iterable[cirq.Qid]:
        return [*self.data, self.syn[i]]

    def round_detectors(self, lookback: int = 1) -> stim.Circuit:
        """These are after each round."""
        circuit = stim.Circuit()
        for i in range(len(self.measured_observables)):
            # -len(stab_gens) because we're looking at the previous round
            circuit += stim.Circuit(
                f"DETECTOR rec[{i - len(self.measured_observables)}] rec[{i - (1 + lookback) * len(self.measured_observables)}]"
            )
        return circuit

    def reset_all(self) -> stim.Circuit:
        return stim.Circuit(
            f"R {' '.join(str(self.qid_map[q]) for q in self.data + self.syn)}"
        )


def star_shaped_ordering():
    """Provides the star shaped measurement order.
    This ordering generates a distinguishable fault set up to weight 4 fault combinations.
    """

    orderings = []

    # weight 4 stabilizer generatrs
    for i in range(7):
        offset = i * 7
        orderings.append([offset + 0, offset + 1, offset + 2, offset + 6])
        orderings.append([offset + 2, offset + 3, offset + 4, offset + 6])
        orderings.append([offset + 4, offset + 5, offset + 0, offset + 6])

    # fmt: off
    orderings.append([                
            1, 12, 17, 47, 
            2, 7, 18, 42, 
            3, 8, 19, 43,
    ])
     
    orderings.append([
            15, 26, 31, 43, 
            16, 21, 32, 44, 
            17, 22, 33, 45,
    ])
    orderings.append([
            29, 40, 3, 45, 
            30, 35, 4, 46, 
            31, 36, 5, 47,
    ])
    # fmt: on

    colorings = {
        "red": [2, 4, 6, 9, 13, 17, 19, 21],
        "green": [1, 5, 8, 10, 12, 15, 20, 22],
        "blue": [0, 3, 7, 11, 14, 16, 18, 23],
    }
    return orderings, colorings

def star_shaped_ordering_m1():
    """Provides the star shaped measurement order.
    This ordering generates a distinguishable fault set up to weight 4 fault combinations.
    """
    print("m1")

    orderings = []

    # weight 4 stabilizer generatrs
    orderings.append([0, 1, 2, 6])
    orderings.append([2, 3, 4, 6])
    orderings.append([4, 5, 0, 6])

    colorings = {
        "red": [2, 4, 6],
        "green": [1, 5],
        "blue": [0, 3],
    }
    return orderings, colorings


def star_shaped_ordering_m3():
    """Provides the star shaped measurement order.
    This ordering generates a distinguishable fault set up to weight 4 fault combinations.
    """

    orderings = []

    # weight 4 stabilizer generatrs
    for i in range(7):
        large_offset = i * 49
        for j in range(7):
            offset = j * 7 + large_offset
            orderings.append([offset + 0, offset + 1, offset + 2, offset + 6])
            orderings.append([offset + 2, offset + 3, offset + 4, offset + 6])
            orderings.append([offset + 4, offset + 5, offset + 0, offset + 6])

        #weight 12 stabilizer generators
        stab12_1 = [                
                1, 12, 17, 47, 
                2, 7, 18, 42, 
                3, 8, 19, 43,
        ]

        stab12_2 = [
                15, 26, 31, 43, 
                16, 21, 32, 44, 
                17, 22, 33, 45,
        ]
        stab12_3 = [
                29, 40, 3, 45, 
                30, 35, 4, 46, 
                31, 36, 5, 47,
        ]


        orderings.append([x+large_offset for x in stab12_1])
        orderings.append([x+large_offset for x in stab12_2])
        orderings.append([x+large_offset for x in stab12_3])
    
    #weight 36 stabelizer generators
    stab36_1 = [10,59,122,332,
                11,58,123,333,
                12,57,124,334,
                19,50,131,299,
                14,49,126,294,
                15,54,127,295,
                22,89,134,302,
                23,87,135,303,
                24,88,136,304]
    
    stab36_2 = [108,185,220,304,
                109,186,221,305,
                110,187,222,306,
                117,152,229,313,
                112,147,224,308,
                113,148,225,309,
                120,155,232,316,
                121,156,233,317,
                122,157,234,318,]
    
    stab36_3 = [206,283,24,318,
                207,284,25,319,
                208,285,26,320,
                215,250,33,327,
                210,245,28,322,
                211,246,29,323,
                218,253,36,330,
                219,254,37,331,
                220,255,38,332,]
    
    orderings.append(stab36_1)
    orderings.append(stab36_2)
    orderings.append(stab36_3)


    colorings = {
        "red": [2, 4, 6, 9, 13, 17, 19, 21],
        "green": [1, 5, 8, 10, 12, 15, 20, 22],
        "blue": [0, 3, 7, 11, 14, 16, 18, 23],
    }
    return orderings, colorings



def generate_offline_steane_l2_bare_ancilla(m):

    if m == 1:
        r = 6
        n = 7
        orderings, _ = star_shaped_ordering_m1()
    elif m == 2:
        r = 48
        n = 49
        orderings, _ = star_shaped_ordering()
    elif m == 3:
        r = 342
        n = 343
        orderings, _ = star_shaped_ordering_m3()
    else:
        raise ValueError("Invalid value of m")

    qs = cirq.LineQubit.range(n)
    # r ancilla for the plaquettes and one for the logical observable
    anc = cirq.LineQubit.range(n, n + r + 1)

    stabilizer_gens = [OrderedObservable("X", g) for g in orderings] + [
        OrderedObservable("Z", g) for g in orderings
    ]

    experiment = BareAncillaOffline(
        stabilizer_generators=stabilizer_gens,
        logical_observables=['Z'*n],
        data=qs,
        syn=anc,
        n_rounds=1,
    )

    circuit = experiment.generate()
    return circuit

def get_concat_steane_code(m):
    #read whaterver.stim file and make stim circuit out of it
    circuit = stim.Circuit.from_file("our_own_2.stim")
    return StimCodeCircuit(stim_circuit=circuit)
    #circuit = filter_ticks(generate_offline_steane_l2_bare_ancilla(m))
    #print(circuit)
    #return StimCodeCircuit(stim_circuit=circuit)







if __name__ == "__main__":
    circuit = get_concat_steane_code(3)
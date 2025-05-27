import stim 
from qiskit_qec.circuits import StimCodeCircuit

from .steane_circuit_parts import concat_steane_end, concat_steane_round, concat_steane_start

import traceback

def star_shaped_ordering_m1():
    """Provides the star shaped measurement order.
    This ordering generates a distinguishable fault set up to weight 4 fault combinations.
    """

    orderings = []

    # weight 4 stabilizer generatrs
    orderings.append([0, 1, 2, 6])
    orderings.append([2, 3, 4, 6])
    orderings.append([4, 5, 0, 6])


    return orderings



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

    return orderings

def build_big_concat_steane_circuit():
    circuit = stim.Circuit()
    #first encoding of the circuit
    orderings = star_shaped_ordering_m3()
    for ordering in orderings:
        #MPP for each of the entries
        string = "MPP "
        for j in ordering:
            string += "X" +str(j)+"*"
        #remove last element of string
        string = string[:-1]
        string += "\n"
        circuit += stim.Circuit(f"{string}")
        string = ""

    for i in range(343, 514):
        circuit += stim.Circuit(f"RX {i}\n")
    
    for i in range(514, 686): #Want to reset 685 as well because last ancilla
        circuit += stim.Circuit(f"R {i}\n")

    #X
    for ordering, i in zip(orderings, range(343,514)):
        string = "CX "
        for j in ordering:
            string += str(i) + " " + str(j) + " "
        circuit += stim.Circuit(f"{string}\n")
        string = ""
    #Z
    for ordering, i in zip(orderings, range(343,514)):
        string = "CX "
        for j in ordering:
            string += str(j) + " " + str(i) + " "
        circuit += stim.Circuit(f"{string}\n")
        string = ""
    
    #last ancilla
    string = "CX "
    for i in range(343):
        string += str(685) + " " + str(i) + " "
    circuit += stim.Circuit(f"{string}\n")
    string = ""

    #add measurements
    for i in range(343, 514):
        circuit += stim.Circuit(f"MRX {i}\n")
    
    for i in range(514, 686): #Want to reset 685 as well because last ancilla
        circuit += stim.Circuit(f"MR {i}\n")

    #now we need to apply detectors first 171 detectors from -2 to -172
    for i in range(172,1, -1):
        circuit += stim.Circuit(f"DETECTOR rec[{0-i}]\n")
    
    for i in range(343,172, -1):
        circuit += stim.Circuit(f"DETECTOR rec[{0-i}] rec[{0-i-171}]\n")
    
    #Add observable
    circuit += stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-1]\n")

    return circuit

def multiple_round_steane_code(rounds):
    circuit = stim.Circuit()

    for i in range(rounds):
        for ordering in star_shaped_ordering_m1():
            #MPP for each of the entries
            string = "MPP "
            for j in ordering:
                string += "X" + str(j) + "*"
            #remove last element of string
            string = string[:-1]
            string += "\n"
            circuit += stim.Circuit(f"{string}")
            string = ""

        for i in range(7,10):
            circuit += stim.Circuit(f"RX {i}\n")
        
        for i in range(10,13): #Want to reset 14 as well because last ancilla
            circuit += stim.Circuit(f"R {i}\n")

        for ordering, i in zip(star_shaped_ordering_m1(), range(7,11)):
            string = "CX "
            for j in ordering:
                string += str(i) + " " + str(j) + " "
            circuit += stim.Circuit(f"{string}\n")
            string = ""
        
        for ordering, i in zip(star_shaped_ordering_m1(), range(10,13)):
            string = "CX "
            for j in ordering:
                string += str(j) + " " + str(i) + " "
            circuit += stim.Circuit(f"{string}\n")
            string = ""
        
        #last ancilla
        string = "CX "
        for i in range(7):
            string += str(13) + " " + str(i) + " "
        circuit += stim.Circuit(f"{string}\n")
        string = ""

        #add measurements
        for i in range(7, 10):
            circuit += stim.Circuit(f"MRX {i}\n")
        
        for i in range(10, 13): #Want to reset 14 as well because last ancilla
            circuit += stim.Circuit(f"MR {i}\n")
        
        #now we need to apply detectors first 6 detectors from -2 to -7
        for i in range(4, 1, -1):
            circuit += stim.Circuit(f"DETECTOR rec[{0-i}]\n")
        
        count = 5
        for i in range(5, 8):
            circuit += stim.Circuit(f"DETECTOR rec[{0-i}] rec[{0-i-count}]\n")
            count = count - 2
        
        #Add observable
    circuit += stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-1]\n")
    return circuit

def multiple_round_steane_code_own(rounds):
    circuit = stim.Circuit('''# ——— Steane logical |0> prep ———
# (this MPP‐based prep records 3 bits, which we will never reference in any DETECTOR)

MPP X3*X4*X5*X6
MPP X1*X2*X5*X6
MPP X0*X2*X4*X6
TICK

# ——— Round 1: ancilla prep ———
# RX is “reset to |+>” (no rec), R is “reset to |0>” (no rec)
RX 7 8 9
R  10 11 12
TICK

# ——— Round 1: entangle X‐synd and Z‐synd ancillas ———
CX 7 6 8 5
TICK
CX 7 2 8 1
TICK
CX 7 4 8 6
TICK
CX 7 0 8 2 9 5
TICK
CX 9 6
TICK
CX 9 3
TICK
CX 9 4
TICK

# CNOTs for Z stabilizers
CX 6 12
TICK
CX 3 12
TICK
CX 4 12
TICK
CX 6 10 5 11
TICK
CX 2 10 1 11
TICK
CX 4 10 6 11
TICK
CX 0 10 2 11 5 12
TICK

# ——— Round 1: measure ancillas into rec bits ———
MRX 7 8 9      # X‐basis measures → rec bits #4,5,6 of this “measurement block”
MR  10 11 12   # Z‐basis measures → rec bits #1,2,3 of this block
TICK

DETECTOR rec[-6] rec[-7]        
DETECTOR rec[-5] rec[-8]        
DETECTOR rec[-4] rec[-9]       
DETECTOR rec[-3]         # Z‐stab on (3,4,5,6)
DETECTOR rec[-2]         # Z‐stab on (1,2,5,6)
DETECTOR rec[-1]         # Z‐stab on (0,2,4,6)''')
    


    for i in range(rounds-1):
        round_circuit = stim.Circuit('''CX 7 6 8 5
TICK
CX 7 2 8 1
TICK
CX 7 4 8 6
TICK
CX 7 0 8 2 9 5
TICK
CX 9 6
TICK
CX 9 3
TICK
CX 9 4
TICK

# CNOTs for Z stabilizers
CX 6 12
TICK
CX 3 12
TICK
CX 4 12
TICK
CX 6 10 5 11
TICK
CX 2 10 1 11
TICK
CX 4 10 6 11
TICK
CX 0 10 2 11 5 12
TICK

# ——— Round 1: measure ancillas into rec bits ———
MRX 7 8 9      # X‐basis measures → rec bits #4,5,6 of this “measurement block”
MR  10 11 12   # Z‐basis measures → rec bits #1,2,3 of this block
TICK

DETECTOR rec[-6] rec[-12]        
DETECTOR rec[-5] rec[-11]        
DETECTOR rec[-4] rec[-10]       
DETECTOR rec[-3] rec[-9]        # Z‐stab on (3,4,5,6)
DETECTOR rec[-2] rec[-8]        # Z‐stab on (1,2,5,6)
DETECTOR rec[-1] rec[-7]        # Z‐stab on (0,2,4,6)
''')
        #add round_circuit to circuit
        circuit += round_circuit
    
    #add observable
    circuit += stim.Circuit('''CX 13 0  13 1  13 2  13 3  13 4  13 5  13 6
MR 13
OBSERVABLE_INCLUDE(0) rec[-1]''')

    return circuit

def concat_steane_multiple_cycles(rounds=1):
    circuit = stim.Circuit(concat_steane_start)
    for i in range(rounds-1):
        circuit += stim.Circuit(concat_steane_round)
    circuit += stim.Circuit(concat_steane_end)
    return circuit


def get_concat_steane_code(m, rounds=1):
    #read whaterver.stim file and make stim circuit out of it
    if m==2:
        #circuit = stim.Circuit.from_file("codes/concat-steane_multiple_cycles.stim")
        circuit = concat_steane_multiple_cycles(rounds)
    elif m==1:
        #circuit = stim.Circuit.from_file("codes/multiple_rounds.stim")
        circuit = multiple_round_steane_code_own(rounds)
    elif m==3:
        circuit = stim.Circuit.from_file("codes/concat_3.stim")
    
    print(circuit)
    try: 
        s = StimCodeCircuit(stim_circuit=circuit)
    except Exception as e:
        print(f"Error while creating StimCodeCircuit: {e}")
        print(traceback.format_exc())
        return None
    print("Do we manage?")
    return s


if __name__ == "__main__":
    #circuit = build_big_concat_steane_circuit()
    #circuit = get_concat_steane_code(3)

    circuit = multiple_round_steane_code_own(2)
    print(circuit)
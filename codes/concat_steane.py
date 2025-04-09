import stim 
from qiskit_qec.circuits import StimCodeCircuit

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

def build_big_concat_steane_circuit():
    circuit = stim.Circuit()
    #first encoding of the circuit
    orderings = star_shaped_ordering_m3()[0]
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

def get_concat_steane_code(m):
    #read whaterver.stim file and make stim circuit out of it
    if m==2:
        circuit = stim.Circuit.from_file("codes/concat_steane_own.stim")
    elif m==1:
        circuit = stim.Circuit.from_file("codes/gidney.stim")
    elif m==3:
        circuit = stim.Circuit.from_file("codes/concat_3.stim")
    return StimCodeCircuit(stim_circuit=circuit)


if __name__ == "__main__":
    circuit = build_big_concat_steane_circuit()
    #circuit = get_concat_steane_code(3)
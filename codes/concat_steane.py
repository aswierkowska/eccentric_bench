import stim 
from qiskit_qec.circuits import StimCodeCircuit

from .steane_circuit_parts import concat_steane_end, concat_steane_round, concat_steane_start, normal_steane_start, normal_steane_round, normal_steane_end

import traceback

def ordering_normal_stean_code():
    orderings = []
    orderings.append([0, 1, 2, 6])
    orderings.append([2, 3, 4, 6])
    orderings.append([4, 5, 0, 6])

    return orderings

def ordering_concat_steane_code():
    """Provides the star shaped measurement order.
    This ordering generates a distinguishable fault set up to weight 4 fault combinations.
    """
    orderings = []
    
    # weight 4 stabilizer generators
    for j in range(7):
        offset = j * 7
        orderings.append([offset + 0, offset + 1, offset + 2, offset + 6])
        orderings.append([offset + 2, offset + 3, offset + 4, offset + 6])
        orderings.append([offset + 4, offset + 5, offset + 0, offset + 6])

    
    # weight 12 stabilizer generators
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
    orderings.append(stab12_1)
    orderings.append(stab12_2)
    orderings.append(stab12_3)

    return orderings

def ordering_extended_steane_code():
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

def normal_steane_circuit_f(rounds = 0):
    circuit = stim.Circuit()
    #first encoding of the circuit
    orderings = ordering_normal_stean_code()
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
        circuit += stim.Circuit("TICK")

    for i in range(7, 10):
        circuit += stim.Circuit(f"R {i}\n")

    for i in range(7, 10):
        circuit += stim.Circuit(f"H {i}\n")
    
    for i in range(10, 14): #Want to reset 20 as well because last ancilla
        circuit += stim.Circuit(f"R {i}\n")

    #X
    for ordering, i in zip(orderings, range(7,10)):
        string = "CX "
        for j in ordering:
            string += str(i) + " " + str(j) + " "
        circuit += stim.Circuit(f"{string}\n")
        string = ""
    circuit += stim.Circuit("TICK")
 
    #Z
    for ordering, i in zip(orderings, range(10,13)):
        string = "CX "
        for j in ordering:
            string += str(j) + " " + str(i) + " "
        circuit += stim.Circuit(f"{string}\n")
        string = ""
    circuit += stim.Circuit("TICK")

    #add measurements
    for i in range(7, 10):
        circuit += stim.Circuit(f"H {i}\n")
    for i in range(7, 10):
        circuit += stim.Circuit(f"MR {i}\n")

    circuit += stim.Circuit("TICK")
    
    for i in range(10, 13): #Want to reset 20 as well because last ancilla
        circuit += stim.Circuit(f"MR {i}\n")

    circuit += stim.Circuit("TICK")
    
    for i in range(6,3,-1):
        circuit += stim.Circuit(f"DETECTOR rec[{0-i}] rec[{0-i-3}]\n")
    
    for i in range(3,0, -1):
        circuit += stim.Circuit(f"DETECTOR rec[{0-i}]\n")

    circuit += stim.Circuit("TICK")
    #add cycles
    circuit = add_cycles_normal_steane_code(circuit, orderings, rounds-1)

    #last ancilla
    string = "CX "
    for i in range(7):
        string += str(13) + " " + str(i) + " "
    circuit += stim.Circuit(f"{string}\n")
    string = ""
    circuit += stim.Circuit("TICK")
    circuit += stim.Circuit("MR 13\n")
    circuit += stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-1]\n")
    return circuit

def add_cycles_normal_steane_code(circuit, orderings, rounds):
    for r in range(rounds):
        #X
        for i in range(7, 10):
            circuit += stim.Circuit(f"H {i}\n")
        for ordering, i in zip(orderings, range(7,10)):
            string = "CX "
            for j in ordering:
                string += str(i) + " " + str(j) + " "
            circuit += stim.Circuit(f"{string}\n")
            string = ""
        circuit += stim.Circuit("TICK")
   
        #Z
        for ordering, i in zip(orderings, range(10,13)):
            string = "CX "
            for j in ordering:
                string += str(j) + " " + str(i) + " "
            circuit += stim.Circuit(f"{string}\n")
            string = ""
        circuit += stim.Circuit("TICK")

        #add measurements
        for i in range(7, 10):
            circuit += stim.Circuit(f"H {i}\n")
        for i in range(7, 10):
            circuit += stim.Circuit(f"MR {i}\n")

        circuit += stim.Circuit("TICK")
        
        for i in range(10, 13):
            circuit += stim.Circuit(f"MR {i}\n")

        circuit += stim.Circuit("TICK")

        for i in range(6,0,-1):
            circuit += stim.Circuit(f"DETECTOR rec[{0-i}] rec[{0-i-6}]\n")
        
    return circuit

def concat_steane_circuit_f(rounds = 0):
    circuit = stim.Circuit()

    orderings = ordering_concat_steane_code()
    #first encoding of the circuit
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
        circuit += stim.Circuit("TICK")
    
    for i in range(49, 73):
        circuit += stim.Circuit(f"R {i}\n")
    
    for i in range(49, 73):
        circuit += stim.Circuit(f"H {i}\n")
    
    for i in range(73, 98):
        circuit += stim.Circuit(f"R {i}\n")
    
    #X
    for ordering, i in zip(orderings, range(49,73)):
        string = "CX "
        for j in ordering:
            string += str(i) + " " + str(j) + " "
        circuit += stim.Circuit(f"{string}\n")
        circuit += stim.Circuit("TICK")
        string = ""
        
    circuit += stim.Circuit("TICK")

    #Z
    for ordering, i in zip(orderings, range(73,97)):
        string = "CX "
        for j in ordering:
            string += str(j) + " " + str(i) + " "
        circuit += stim.Circuit(f"{string}\n")
        circuit += stim.Circuit("TICK")
        string = ""
        
    circuit += stim.Circuit("TICK")

    #add measurements
    for i in range(49, 73):
        circuit += stim.Circuit(f"H {i}\n")
    for i in range(49, 73):
        circuit += stim.Circuit(f"MR {i}\n")
    circuit += stim.Circuit("TICK")

    for i in range(73, 97):
        circuit += stim.Circuit(f"MR {i}\n")
    circuit += stim.Circuit("TICK")

    for i in range(48, 24, -1):
        circuit += stim.Circuit(f"DETECTOR rec[{0-i}] rec[{0-i-24}]\n")
    for i in range(24, 0, -1):
        circuit += stim.Circuit(f"DETECTOR rec[{0-i}]\n")
    
    circuit += stim.Circuit("TICK")
    #add cycles
    circuit = add_cycles_concat_steane_code(circuit, orderings, rounds-1)

    #last ancilla
    string = "CX "
    for i in range(49):
        string += str(97) + " " + str(i) + " "
    circuit += stim.Circuit(f"{string}\n")
    string = ""

    circuit += stim.Circuit("TICK")
    circuit += stim.Circuit("MR 97\n")
    circuit += stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-1]\n")

    return circuit

def add_cycles_concat_steane_code(circuit, orderings, rounds):
    for r in range(rounds):
        #X
        for i in range(49, 73):
            circuit += stim.Circuit(f"H {i}\n")
        for ordering, i in zip(orderings, range(49,73)):
            string = "CX "
            for j in ordering:
                string += str(i) + " " + str(j) + " "
            circuit += stim.Circuit(f"{string}\n")
            string = ""
        circuit += stim.Circuit("TICK")

        #Z
        for ordering, i in zip(orderings, range(73,97)):
            string = "CX "
            for j in ordering:
                string += str(j) + " " + str(i) + " "
            circuit += stim.Circuit(f"{string}\n")
            string = ""
        circuit += stim.Circuit("TICK")

        #add measurements
        for i in range(49, 73):
            circuit += stim.Circuit(f"H {i}\n")
        for i in range(49, 73):
            circuit += stim.Circuit(f"MR {i}\n")

        circuit += stim.Circuit("TICK")
        
        for i in range(73, 97):
            circuit += stim.Circuit(f"MR {i}\n")

        circuit += stim.Circuit("TICK")

        for i in range(48,0,-1):
            circuit += stim.Circuit(f"DETECTOR rec[{0-i}] rec[{0-i-48}]\n")
        
    return circuit

def extended_steane_circuit(rounds=0):
    circuit = stim.Circuit()
    #first encoding of the circuit
    orderings = ordering_extended_steane_code()
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
        circuit += stim.Circuit("TICK")

    for i in range(343, 514):
        circuit += stim.Circuit(f"R {i}\n")

    for i in range(343, 514):
        circuit += stim.Circuit(f"H {i}\n")
    
    for i in range(514, 686): #Want to reset 685 as well because last ancilla
        circuit += stim.Circuit(f"R {i}\n")

    #X
    for ordering, i in zip(orderings, range(343,514)):
        string = "CX "
        for j in ordering:
            string += str(i) + " " + str(j) + " "
        circuit += stim.Circuit(f"{string}\n")
        string = ""
    circuit += stim.Circuit("TICK")
 
    #Z
    for ordering, i in zip(orderings, range(514,685)):
        string = "CX "
        for j in ordering:
            string += str(j) + " " + str(i) + " "
        circuit += stim.Circuit(f"{string}\n")
        string = ""
    circuit += stim.Circuit("TICK")

    #add measurements
    for i in range(343, 514):
        circuit += stim.Circuit(f"H {i}\n")
    for i in range(343, 514):
        circuit += stim.Circuit(f"MR {i}\n")

    circuit += stim.Circuit("TICK")
    
    for i in range(514, 685): #Want to reset 685 as well because last ancilla
        circuit += stim.Circuit(f"MR {i}\n")

    circuit += stim.Circuit("TICK")
    
    for i in range(342,171, -1):
        circuit += stim.Circuit(f"DETECTOR rec[{0-i}] rec[{0-i-171}]\n")

    for i in range(171,0, -1):
        circuit += stim.Circuit(f"DETECTOR rec[{0-i}]\n")

    circuit += stim.Circuit("TICK")
    circuit = add_cycles_extended_steane_circuit(circuit, orderings, rounds)

    #last ancilla
    string = "CX "
    for i in range(343):
        string += str(685) + " " + str(i) + " "
    circuit += stim.Circuit(f"{string}\n")
    string = ""

    circuit += stim.Circuit("TICK")
    circuit += stim.Circuit("MR 685\n")
    circuit += stim.Circuit("OBSERVABLE_INCLUDE(0) rec[-1]\n")

    return circuit

def add_cycles_extended_steane_circuit(circuit, orderings, rounds):
    for r in range(rounds):
        #X
        for i in range(343, 514):
            circuit += stim.Circuit(f"H {i}\n")
        for ordering, i in zip(orderings, range(343,514)):
            string = "CX "
            for j in ordering:
                string += str(i) + " " + str(j) + " "
            circuit += stim.Circuit(f"{string}\n")
            string = ""
        circuit += stim.Circuit("TICK")
   
        #Z
        for ordering, i in zip(orderings, range(514,685)):
            string = "CX "
            for j in ordering:
                string += str(j) + " " + str(i) + " "
            circuit += stim.Circuit(f"{string}\n")
            string = ""
        circuit += stim.Circuit("TICK")

        #add measurements
        for i in range(343, 514):
            circuit += stim.Circuit(f"H {i}\n")
        for i in range(343, 514):
            circuit += stim.Circuit(f"MR {i}\n")

        circuit += stim.Circuit("TICK")
        
        for i in range(514, 685): #Want to reset 685 as well because last ancilla
            circuit += stim.Circuit(f"MR {i}\n")

        circuit += stim.Circuit("TICK")

        for i in range(342,0, -1):
            circuit += stim.Circuit(f"DETECTOR rec[{0-i}] rec[{0-i-342}]\n")
    return circuit


def normal_steane_circuit(rounds):
    circuit = stim.Circuit(normal_steane_start)
    for i in range(rounds-1):
        round_circuit = stim.Circuit(normal_steane_round)
        circuit += round_circuit  
    circuit += stim.Circuit(normal_steane_end)
    return circuit

def concat_steane_circuit(rounds=1):
    circuit = stim.Circuit(concat_steane_start)
    for i in range(rounds-1):
        circuit += stim.Circuit(concat_steane_round)
    circuit += stim.Circuit(concat_steane_end)
    return circuit


def get_concat_steane_code(m, rounds=1,basis="Z"):
    if m==1:
        circuit = normal_steane_circuit_f(rounds)
    elif m==2:
        circuit = concat_steane_circuit_f(rounds)
    elif m==3:
        circuit = extended_steane_circuit(rounds)
    try: 
        s = StimCodeCircuit(stim_circuit=circuit)
    except Exception as e:
        print(f"Error while creating StimCodeCircuit: {e}")
        print(traceback.format_exc())
        return None
    return s


if __name__ == "__main__":
    circuit = concat_steane_circuit_f(2)
    circuit_2 = concat_steane_circuit(2)
    print(circuit)
    print("--------------------------------------------------------------------------")
    print(circuit_2)

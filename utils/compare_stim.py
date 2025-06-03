import sys

def normalize_stim_line(line):
    """Expands gates with multiple targets into separate lines."""
    tokens = line.strip().split()
    if not tokens:
        return []

    gate = tokens[0]
    targets = tokens[1:]

    # Skip comments or unsupported lines
    if gate.startswith('#') or gate == '':
        return []

    # Handle gates like 'CX 0 1 1 2' → [('CX', [0, 1]), ('CX', [1, 2])]
    if gate in {"CX", "CZ", "SWAP"}:
        if len(targets) % 2 != 0:
            raise ValueError(f"Invalid 2-target gate: {line}")
        return [f"{gate} {targets[i]} {targets[i+1]}" for i in range(0, len(targets), 2)]
    
    # Single-qubit gates (H, M, etc.)
    return [f"{gate} {target}" for target in targets]

def normalize_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    normalized = []
    for line in lines:
        normalized.extend(normalize_stim_line(line))
    
    return sorted(normalized)

def compare_files(file1, file2):
    norm1 = normalize_file(file1)
    norm2 = normalize_file(file2)

    set1 = set(norm1)
    set2 = set(norm2)

    only_in_1 = sorted(set1 - set2)
    only_in_2 = sorted(set2 - set1)

    if not only_in_1 and not only_in_2:
        print("Files are equivalent after normalization.")
    else:
        if only_in_1:
            print(f"\nLines only in {file1}:")
            for line in only_in_1:
                print("  " + line)
        if only_in_2:
            print(f"\nLines only in {file2}:")
            for line in only_in_2:
                print("  " + line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_stim_files.py file1.stim file2.stim")
        sys.exit(1)

    compare_files(sys.argv[1], sys.argv[2])

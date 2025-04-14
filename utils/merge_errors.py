import re
import sys

def remove_duplicate_errors(file_path):
    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    seen = set()
    unique_lines = []

    for line in lines:
        match = re.match(r'^\S+\s+\S+\s+-\s+\S+\s+-\s+(.*)', line)
        if match:
            error_message = match.group(1)
            if error_message not in seen:
                unique_lines.append(line)
                seen.add(error_message)

    with open(file_path, 'w') as outfile:
        outfile.writelines(unique_lines)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 merge_error.py path/to/file.log")
        sys.exit(1)

    file_path = sys.argv[1]
    remove_duplicate_errors(file_path)

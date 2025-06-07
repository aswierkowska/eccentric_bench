import sys

def filter_lines(input_file, output_file):
    keywords = ("ERROR", "DEPOLARIZE")
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if not any(keyword in line for keyword in keywords):
                outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <filename1> <filename2>")
        sys.exit(1)

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    filter_lines(filename1, filename2)

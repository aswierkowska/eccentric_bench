def replace_numbers_in_file(file_path):
    replacements = {
        '114': '58',
        '192': '116',
        '290': '193',
        '408': '292',
        '546': '410'
    }
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Replace all occurrences according to the replacements dict
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Write the modified content back to the same file (in-place)
    with open(file_path, 'w') as file:
        file.write(content)

if __name__ == "__main__":
    file_path = input("Enter the path of the file to modify: ")
    replace_numbers_in_file(file_path)
    print("Replacements done in-place!")
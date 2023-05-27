import json

input_file = "prompt - Copy.json"
output_file = "prompt-new.json"

# Read the input file
with open(input_file, "r") as file:
    data = json.load(file)

# Write each dictionary in a new line in the output file
with open(output_file, "w") as file:
    for item in data:
        # Convert the dictionary to a string
        item_str = json.dumps(item)
        # Write the string to the file with a newline character
        file.write(item_str + "\n")

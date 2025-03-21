import toml
import sys
import os

def generate_requirements(input_dir, output_file):
    # Construct full paths
    pyproject_path = os.path.join(input_dir, "pyproject.toml")

    # Check if the input file exists
    if not os.path.isfile(pyproject_path):
        print(f"Error: The file 'pyproject.toml' does not exist in the directory '{input_dir}'.")
        sys.exit(1)

    # Load pyproject.toml
    try:
        with open(pyproject_path, "r") as f:
            pyproject = toml.load(f)
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
        sys.exit(1)

    # Extract dependencies
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    if not dependencies:
        print("No dependencies found in pyproject.toml.")
        sys.exit(0)

        # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to requirements.txt
    try:
        with open(output_file, "w") as f:
            for dep in dependencies:
                f.write(f"{dep}\n")
        print(f"Requirements written to '{output_file}'")
    except Exception as e:
        print(f"Error writing requirements.txt: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_directory> <outfile>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    generate_requirements(input_dir, output_file)

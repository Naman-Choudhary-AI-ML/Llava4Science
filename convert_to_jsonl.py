import json
import argparse


def convert_to_jsonl(input_file, output_file):
    """
    Converts a JSON file to JSONL format.
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSONL file.
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            data = json.load(infile)
            if isinstance(data, list):  # JSON is a list of objects
                for item in data:
                    outfile.write(json.dumps(item) + '\n')
            else:
                raise ValueError("Expected a list of objects in the input JSON file.")
        print(f"Converted {input_file} to {output_file} successfully.")
    except Exception as e:
        print(f"Error converting {input_file} to JSONL: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON file to JSONL format.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()

    convert_to_jsonl(args.input_file, args.output_file)

import hashlib
import os
from collections import Counter

def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    # First pass: count hashed lines across all files
    line_counts = Counter()
    for file_path in input_files:
        with open(file_path) as file:
            content = file.readlines()
            hashed_content = [hashlib.sha256(line.encode()).hexdigest() for line in content]
            line_counts.update(hashed_content)

    # Second pass: write unique lines to output files
    for file_path in input_files:
        with open(file_path) as file:
            content = file.readlines()
            hashed_content = [hashlib.sha256(line.encode()).hexdigest() for line in content]
            unique_lines = [
                line for line, hashed in zip(content, hashed_content) if line_counts[hashed] == 1
            ]
            output_path = os.path.join(output_directory, os.path.basename(file_path))
            with open(output_path, "w") as output_file:
                output_file.writelines(unique_lines)
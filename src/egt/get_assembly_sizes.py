#!/usr/bin/env python3
import os
import json
import csv

def extract_assembly_info(jsonl_file):
    with open(jsonl_file, 'r') as file:
        data = json.load(file)

        accession = data.get("accession")
        total_sequence_length = data.get("assemblyStats", {}).get("totalSequenceLength")

        return accession, total_sequence_length

def process_directory(directory):
    results = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                accession, total_sequence_length = extract_assembly_info(file_path)
                results.append((accession, total_sequence_length))

    return results

def write_to_tsv(output_file, data):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["Assembly Accession", "Total Sequence Length"])
        writer.writerows(data)

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Summarize assembly sizes from a directory tree of NCBI datasets .jsonl files.",
    )
    parser.add_argument("--genomes-dir", required=True,
                        help="Directory to recursively search for assembly *.jsonl files.")
    parser.add_argument("--output", default="output_assembly_sizes.tsv",
                        help="Output TSV path (default: %(default)s).")
    args = parser.parse_args(argv)
    results = process_directory(args.genomes_dir)
    write_to_tsv(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

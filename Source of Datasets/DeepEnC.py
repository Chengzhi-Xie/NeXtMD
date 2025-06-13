#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

def parse_and_collect(files):
    """
    Parse multiple FASTA files to extract (label, sequence) pairs,
    and unify labels as '0' or '1' (strings).
    """
    entries = []
    for filepath in files:
        if not os.path.isfile(filepath):
            print(f"Warning: file {filepath} not found, skipping.", file=sys.stderr)
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            label = None
            seq_parts = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    # Upon encountering a new header, save the previous entry
                    if label is not None:
                        entries.append((label, ''.join(seq_parts)))
                    header = line
                    # parse the label
                    if 'Positive' in header:
                        label = '1'
                    elif 'Negative' in header:
                        label = '0'
                    else:
                        # for headers like >seq0|1|testing, take the middle number
                        parts = header[1:].split('|')
                        if len(parts) >= 2 and parts[1].isdigit():
                            label = parts[1]
                        else:
                            raise ValueError(f"Cannot parse label number from header: {header}")
                    seq_parts = []
                else:
                    seq_parts.append(line)
            # after finishing file, save the last entry
            if label is not None:
                entries.append((label, ''.join(seq_parts)))
    return entries

def dedupe(entries):
    """
    Remove duplicates: keep only the first occurrence of each sequence.
    """
    seen = set()
    unique = []
    for label, seq in entries:
        if seq not in seen:
            unique.append((label, seq))
            seen.add(seq)
    return unique

def write_merged(entries, output_path):
    """
    Write (label, sequence) pairs to file:
    odd lines: labels (0 or 1)
    even lines: sequences
    """
    with open(output_path, 'w', encoding='utf-8') as out:
        for label, seq in entries:
            out.write(label + '\n')
            out.write(seq   + '\n')
    print(f"Merging complete: wrote {len(entries)} entries (after deduplication) to {output_path}")

if __name__ == '__main__':
    # If needed, this can be modified to read from command-line arguments.
    input_files = [
        'Training dataset.txt',
        'Independent  dataset 1.txt',
        'Independent  dataset 2.txt'
    ]
    output_file = 'merged_sequences_DeepEnC.txt'

    all_entries = parse_and_collect(input_files)
    unique_entries = dedupe(all_entries)
    write_merged(unique_entries, output_file)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

def parse_fasta(filepaths):
    """
    Extract (label, sequence) pairs from multiple FASTA files:
      - Labels ending with '>...|...|pos' -> '1'
      - Labels ending with '>...|...|neg' -> '0'
    """
    entries = []
    for path in filepaths:
        if not os.path.isfile(path):
            print(f"Warning: file {path} not found, skipping.", file=sys.stderr)
            continue

        with open(path, 'r', encoding='utf-8') as f:
            current_label = None
            seq_chunks = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    # before starting a new record, save the previous one
                    if current_label is not None and seq_chunks:
                        entries.append((current_label, ''.join(seq_chunks)))
                    # parse the label
                    header = line[1:]  # remove '>'
                    parts = header.split('|')
                    if len(parts) >= 3:
                        tag = parts[2].lower()
                        if tag == 'pos':
                            current_label = '1'
                        elif tag == 'neg':
                            current_label = '0'
                        else:
                            raise ValueError(f"Unrecognized label polarity: {line}")
                    else:
                        raise ValueError(f"Label format error, expected '...|...|pos' or '...|...|neg': {line}")
                    seq_chunks = []
                else:
                    # sequence line
                    seq_chunks.append(line)

            # after file ends, save the last record
            if current_label is not None and seq_chunks:
                entries.append((current_label, ''.join(seq_chunks)))

    return entries

def dedupe(entries):
    """
    Remove duplicates: keep the first occurrence of each sequence.
    """
    seen = set()
    unique = []
    for label, seq in entries:
        if seq not in seen:
            unique.append((label, seq))
            seen.add(seq)
    return unique

def write_merged(entries, out_path):
    """
    Write (label, sequence) pairs line by line:
      Odd lines: label (0 or 1)
      Even lines: sequence
    """
    with open(out_path, 'w', encoding='utf-8') as out:
        for label, seq in entries:
            out.write(label + '\n')
            out.write(seq   + '\n')
    print(f"Merging complete, wrote {len(entries)} entries (after deduplication) to {out_path}")

if __name__ == '__main__':
    # list of input FASTA files; adjust if needed
    input_files = ['test_dataset.fa', 'training_dataset.fa']
    output_file = 'merged_sequences_BertAIP.txt'

    all_entries = parse_fasta(input_files)
    unique_entries = dedupe(all_entries)
    write_merged(unique_entries, output_file)

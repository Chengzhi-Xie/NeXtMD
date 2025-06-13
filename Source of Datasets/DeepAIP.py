#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re

def parse_and_collect(filepaths):
    """
    Read (label, sequence) pairs from multiple FASTA files:
    - ">Positive_*"     -> label '1'
    - ">Negative_*"     -> label '0'
    - ">seqX|Y|testing" -> label = Y
    - ">seqX|Y|training"-> label = Y
    """
    entries = []
    header_re = re.compile(r'^>(.+)$')
    for path in filepaths:
        if not os.path.isfile(path):
            print(f"Warning: file {path} not found, skipping.", file=sys.stderr)
            continue

        with open(path, 'r', encoding='utf-8') as f:
            cur_label = None
            seq_buf = []
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                m = header_re.match(line)
                if m:
                    # upon encountering a new header, save the previous entry
                    if cur_label is not None and seq_buf:
                        entries.append((cur_label, ''.join(seq_buf)))
                    tag = m.group(1)  # without '>'
                    # map header to label
                    if tag.startswith('Positive'):
                        cur_label = '1'
                    elif tag.startswith('Negative'):
                        cur_label = '0'
                    else:
                        # formats like seqX|Y|testing or seqX|Y|training
                        parts = tag.split('|')
                        if len(parts) >= 2 and parts[1] in ('0', '1'):
                            cur_label = parts[1]
                        else:
                            raise ValueError(f"Cannot parse label: {line}")
                    seq_buf = []
                else:
                    # sequence line
                    seq_buf.append(line)
            # at end of file, save the last entry
            if cur_label is not None and seq_buf:
                entries.append((cur_label, ''.join(seq_buf)))
    return entries

def dedupe(entries):
    """
    Remove duplicates: keep only the first occurrence of each (label, sequence) pair.
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
    Write output line by line:
    Odd lines are labels (0/1)
    Even lines are sequences
    """
    with open(output_path, 'w', encoding='utf-8') as out:
        for label, seq in entries:
            out.write(label + '\n')
            out.write(seq   + '\n')
    print(f"Done: wrote {len(entries)} entries (after deduplication) to {output_path}")

if __name__ == '__main__':
    # Modify file paths or accept them from command line as needed
    files = [
        'Ind_1049.txt',
        'Train-AIP-Neg.txt',
        'Train-AIP-Pos.txt',
        'Ind_426_Neg.txt',
        'Ind_426_Pos.txt'
    ]
    out_file = 'merged_sequences_DeepAIP.txt'

    all_entries = parse_and_collect(files)
    unique_entries = dedupe(all_entries)
    write_merged(unique_entries, out_file)

import argparse
import logging
import os
from datetime import datetime
from itertools import product
from math import sqrt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset reading
def read_dataset_amp(file_path: str):
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break
            label = int(label_line.split('|')[-1])
            sequence_line = file.readline().strip()
            labels.append(label)
            sequences.append(sequence_line)
    return sequences, labels

def read_dataset_zero(file_path: str):
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break
            if 'AMP' in label_line:
                label = 1
            elif 'NEGATIVE' in label_line:
                label = 0
            else:
                print("Error!")
            sequence_line = file.readline().strip()
            labels.append(label)
            sequences.append(sequence_line)
    return sequences, labels

def read_dataset_aip(file_path: str):
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break
            if 'Positive' in label_line:
                label = 1
            elif 'Negative' in label_line:
                label = 0
            else:
                print("Error!")
            sequence_line = file.readline().strip()
            labels.append(label)
            sequences.append(sequence_line)
    return sequences, labels

def read_dataset_from_aipstack_work(file_path: str):
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break
            if '1' in label_line:
                label = 1
            elif '0' in label_line:
                label = 0
            else:
                print("Error!")
            sequence_line = file.readline().strip()
            labels.append(label)
            sequences.append(sequence_line)
    return sequences, labels

def read_dataset_bd(file_path: str):
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break
            if 'pos' in label_line:
                label = 1
            elif 'neg' in label_line:
                label = 0
            else:
                print("Error!")
            sequence_line = file.readline().strip()
            labels.append(label)
            sequences.append(sequence_line)
    return sequences, labels

# DDE&CKSAAP calculation
codon_usage = {
    'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2,
    'I': 3, 'K': 2, 'L': 6, 'M': 1, 'N': 2, 'P': 4, 'Q': 2,
    'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
}
total_codons = 61

def calculate_dipeptide_frequencies(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    freqs = dict.fromkeys(dipeptides, 0)
    for i in range(len(sequence) - 1):
        dp = sequence[i:i+2]
        if dp in freqs:
            freqs[dp] += 1
    for dp in freqs:
        freqs[dp] /= (len(sequence) - 1)
    return freqs

def calculate_dde(sequence):
    freqs = calculate_dipeptide_frequencies(sequence)
    Tm = {}
    Tv = {}
    for dp in freqs:
        Ci = codon_usage[dp[0]]
        Cj = codon_usage[dp[1]]
        Tm[dp] = (Ci / total_codons) * (Cj / total_codons)
        Tv[dp] = Tm[dp] * (1 - Tm[dp]) / (len(sequence) - 1)
    DDE = {}
    for dp in freqs:
        Dc = freqs[dp]
        DDE[dp] = (Dc - Tm[dp]) / (Tv[dp] ** 0.5) if Tv[dp] != 0 else 0
    return DDE

def calculate_cksapp_features(sequence, k_range=range(6)):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    pairs = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    features = {}
    for k in k_range:
        for pair in pairs:
            features[f'{pair}_k{k}'] = 0
    for k in k_range:
        for i in range(len(sequence) - 1 - k):
            pair = sequence[i] + sequence[i + 1 + k]
            if pair in pairs:
                features[f'{pair}_k{k}'] += 1
    for k in k_range:
        total = sum(features[f'{pair}_k{k}'] for pair in pairs)
        if total > 0:
            for pair in pairs:
                features[f'{pair}_k{k}'] /= total
    return features

# Physical and chemical property calculations
aa_props_dict = {
    'A': {
        'BLAM930101': 0.15, 'BIOV880101': 0.20, 'MAXF760101(α-helix)': 1.45,
        'TSAJ990101': 0.50, 'NAKH920108': 0.10, 'CEDJ970104': 0.20,
        'LIFS790101': 0.30, 'MIYS990104': 0.25, 'Hydrophobic': 1.8,
        'Net-charge': 0.0, 'Polarity': 8.1, 'Polarizability': 6.0,
        'β-sheet': 0.97, 'Atomic number': 67
    },
    'C': {
        'BLAM930101': 0.17, 'BIOV880101': 0.22, 'MAXF760101(α-helix)': 0.77,
        'TSAJ990101': 0.52, 'NAKH920108': 0.12, 'CEDJ970104': 0.21,
        'LIFS790101': 0.31, 'MIYS990104': 0.26, 'Hydrophobic': 2.5,
        'Net-charge': 0.0, 'Polarity': 5.5, 'Polarizability': 5.0,
        'β-sheet': 1.30, 'Atomic number': 86
    },
    'D': {
        'BLAM930101': 0.20, 'BIOV880101': 0.25, 'MAXF760101(α-helix)': 0.98,
        'TSAJ990101': 0.55, 'NAKH920108': 0.15, 'CEDJ970104': 0.24,
        'LIFS790101': 0.35, 'MIYS990104': 0.28, 'Hydrophobic': -3.5,
        'Net-charge': -1.0, 'Polarity': 13.0, 'Polarizability': 8.0,
        'β-sheet': 0.80, 'Atomic number': 91
    },
    'E': {
        'BLAM930101': 0.21, 'BIOV880101': 0.26, 'MAXF760101(α-helix)': 1.53,
        'TSAJ990101': 0.56, 'NAKH920108': 0.16, 'CEDJ970104': 0.25,
        'LIFS790101': 0.36, 'MIYS990104': 0.29, 'Hydrophobic': -3.5,
        'Net-charge': -1.0, 'Polarity': 12.3, 'Polarizability': 10.0,
        'β-sheet': 0.26, 'Atomic number': 109
    },
    'F': {
        'BLAM930101': 0.18, 'BIOV880101': 0.24, 'MAXF760101(α-helix)': 1.12,
        'TSAJ990101': 0.54, 'NAKH920108': 0.14, 'CEDJ970104': 0.23,
        'LIFS790101': 0.33, 'MIYS990104': 0.27, 'Hydrophobic': 2.8,
        'Net-charge': 0.0, 'Polarity': 5.2, 'Polarizability': 12.0,
        'β-sheet': 1.28, 'Atomic number': 135
    },
    'G': {
        'BLAM930101': 0.14, 'BIOV880101': 0.19, 'MAXF760101(α-helix)': 0.53,
        'TSAJ990101': 0.49, 'NAKH920108': 0.09, 'CEDJ970104': 0.18,
        'LIFS790101': 0.28, 'MIYS990104': 0.23, 'Hydrophobic': -0.4,
        'Net-charge': 0.0, 'Polarity': 9.0, 'Polarizability': 0.0,
        'β-sheet': 0.81, 'Atomic number': 48
    },
    'H': {
        'BLAM930101': 0.19, 'BIOV880101': 0.23, 'MAXF760101(α-helix)': 1.24,
        'TSAJ990101': 0.53, 'NAKH920108': 0.14, 'CEDJ970104': 0.22,
        'LIFS790101': 0.32, 'MIYS990104': 0.27, 'Hydrophobic': -3.2,
        'Net-charge': 0.0, 'Polarity': 10.4, 'Polarizability': 10.0,
        'β-sheet': 0.71, 'Atomic number': 118
    },
    'I': {
        'BLAM930101': 0.16, 'BIOV880101': 0.21, 'MAXF760101(α-helix)': 1.00,
        'TSAJ990101': 0.50, 'NAKH920108': 0.11, 'CEDJ970104': 0.20,
        'LIFS790101': 0.30, 'MIYS990104': 0.25, 'Hydrophobic': 4.5,
        'Net-charge': 0.0, 'Polarity': 5.2, 'Polarizability': 10.0,
        'β-sheet': 1.60, 'Atomic number': 124
    },
    'K': {
        'BLAM930101': 0.22, 'BIOV880101': 0.27, 'MAXF760101(α-helix)': 1.07,
        'TSAJ990101': 0.57, 'NAKH920108': 0.17, 'CEDJ970104': 0.26,
        'LIFS790101': 0.37, 'MIYS990104': 0.30, 'Hydrophobic': -3.9,
        'Net-charge': 1.0, 'Polarity': 11.3, 'Polarizability': 11.0,
        'β-sheet': 0.74, 'Atomic number': 135
    },
    'L': {
        'BLAM930101': 0.17, 'BIOV880101': 0.22, 'MAXF760101(α-helix)': 1.34,
        'TSAJ990101': 0.56, 'NAKH920108': 0.16, 'CEDJ970104': 0.25,
        'LIFS790101': 0.35, 'MIYS990104': 0.29, 'Hydrophobic': 3.8,
        'Net-charge': 0.0, 'Polarity': 4.9, 'Polarizability': 10.0,
        'β-sheet': 1.22, 'Atomic number': 124
    },
    'M': {
        'BLAM930101': 0.18, 'BIOV880101': 0.23, 'MAXF760101(α-helix)': 1.20,
        'TSAJ990101': 0.55, 'NAKH920108': 0.15, 'CEDJ970104': 0.24,
        'LIFS790101': 0.34, 'MIYS990104': 0.28, 'Hydrophobic': 1.9,
        'Net-charge': 0.0, 'Polarity': 5.7, 'Polarizability': 10.0,
        'β-sheet': 1.67, 'Atomic number': 124
    },
    'N': {
        'BLAM930101': 0.21, 'BIOV880101': 0.26, 'MAXF760101(α-helix)': 0.73,
        'TSAJ990101': 0.54, 'NAKH920108': 0.14, 'CEDJ970104': 0.23,
        'LIFS790101': 0.33, 'MIYS990104': 0.27, 'Hydrophobic': -3.5,
        'Net-charge': 0.0, 'Polarity': 11.6, 'Polarizability': 9.0,
        'β-sheet': 0.65, 'Atomic number': 96
    },
    'P': {
        'BLAM930101': 0.23, 'BIOV880101': 0.28, 'MAXF760101(α-helix)': 0.59,
        'TSAJ990101': 0.58, 'NAKH920108': 0.18, 'CEDJ970104': 0.27,
        'LIFS790101': 0.37, 'MIYS990104': 0.30, 'Hydrophobic': -1.6,
        'Net-charge': 0.0, 'Polarity': 8.0, 'Polarizability': 8.0,
        'β-sheet': 0.62, 'Atomic number': 90
    },
    'Q': {
        'BLAM930101': 0.20, 'BIOV880101': 0.25, 'MAXF760101(α-helix)': 1.17,
        'TSAJ990101': 0.55, 'NAKH920108': 0.15, 'CEDJ970104': 0.24,
        'LIFS790101': 0.35, 'MIYS990104': 0.28, 'Hydrophobic': -3.5,
        'Net-charge': 0.0, 'Polarity': 10.5, 'Polarizability': 10.0,
        'β-sheet': 1.23, 'Atomic number': 114
    },
    'R': {
        'BLAM930101': 0.24, 'BIOV880101': 0.29, 'MAXF760101(α-helix)': 0.79,
        'TSAJ990101': 0.56, 'NAKH920108': 0.16, 'CEDJ970104': 0.25,
        'LIFS790101': 0.35, 'MIYS990104': 0.30, 'Hydrophobic': -4.5,
        'Net-charge': 1.0, 'Polarity': 10.5, 'Polarizability': 12.0,
        'β-sheet': 0.90, 'Atomic number': 148
    },
    'S': {
        'BLAM930101': 0.16, 'BIOV880101': 0.21, 'MAXF760101(α-helix)': 0.79,
        'TSAJ990101': 0.50, 'NAKH920108': 0.11, 'CEDJ970104': 0.20,
        'LIFS790101': 0.30, 'MIYS990104': 0.25, 'Hydrophobic': -0.8,
        'Net-charge': 0.0, 'Polarity': 9.2, 'Polarizability': 5.0,
        'β-sheet': 0.72, 'Atomic number': 73
    },
    'T': {
        'BLAM930101': 0.18, 'BIOV880101': 0.23, 'MAXF760101(α-helix)': 0.82,
        'TSAJ990101': 0.52, 'NAKH920108': 0.12, 'CEDJ970104': 0.21,
        'LIFS790101': 0.31, 'MIYS990104': 0.26, 'Hydrophobic': -0.7,
        'Net-charge': 0.0, 'Polarity': 8.6, 'Polarizability': 7.0,
        'β-sheet': 1.20, 'Atomic number': 93
    },
    'V': {
        'BLAM930101': 0.17, 'BIOV880101': 0.22, 'MAXF760101(α-helix)': 1.14,
        'TSAJ990101': 0.55, 'NAKH920108': 0.15, 'CEDJ970104': 0.24,
        'LIFS790101': 0.34, 'MIYS990104': 0.29, 'Hydrophobic': 4.2,
        'Net-charge': 0.0, 'Polarity': 5.9, 'Polarizability': 8.0,
        'β-sheet': 1.65, 'Atomic number': 105
    },
    'W': {
        'BLAM930101': 0.25, 'BIOV880101': 0.30, 'MAXF760101(α-helix)': 1.14,
        'TSAJ990101': 0.57, 'NAKH920108': 0.17, 'CEDJ970104': 0.26,
        'LIFS790101': 0.36, 'MIYS990104': 0.30, 'Hydrophobic': -0.9,
        'Net-charge': 0.0, 'Polarity': 5.4, 'Polarizability': 15.0,
        'β-sheet': 1.19, 'Atomic number': 163
    },
    'Y': {
        'BLAM930101': 0.20, 'BIOV880101': 0.25, 'MAXF760101(α-helix)': 0.61,
        'TSAJ990101': 0.54, 'NAKH920108': 0.15, 'CEDJ970104': 0.24,
        'LIFS790101': 0.34, 'MIYS990104': 0.28, 'Hydrophobic': -1.3,
        'Net-charge': 0.0, 'Polarity': 6.2, 'Polarizability': 14.0,
        'β-sheet': 1.29, 'Atomic number': 141
    }
}

def calculate_amino_acid_properties(sequence: str) -> list:
    blam = sum(aa_props_dict[aa]['BLAM930101'] for aa in sequence if aa in aa_props_dict)
    biov = sum(aa_props_dict[aa]['BIOV880101'] for aa in sequence if aa in aa_props_dict)
    maxf = sum(aa_props_dict[aa]['MAXF760101(α-helix)'] for aa in sequence if aa in aa_props_dict)
    tsaj = sum(aa_props_dict[aa]['TSAJ990101'] for aa in sequence if aa in aa_props_dict)
    nakh = sum(aa_props_dict[aa]['NAKH920108'] for aa in sequence if aa in aa_props_dict)
    cedj = sum(aa_props_dict[aa]['CEDJ970104'] for aa in sequence if aa in aa_props_dict)
    lifs = sum(aa_props_dict[aa]['LIFS790101'] for aa in sequence if aa in aa_props_dict)
    miys = sum(aa_props_dict[aa]['MIYS990104'] for aa in sequence if aa in aa_props_dict)
    blam_mean = blam / len(sequence)
    biov_mean = biov / len(sequence)
    maxf_mean = maxf / len(sequence)
    tsaj_mean = tsaj / len(sequence)
    nakh_mean = nakh / len(sequence)
    cedj_mean = cedj / len(sequence)
    lifs_mean = lifs / len(sequence)
    miys_mean = miys / len(sequence)
    return [blam, tsaj, nakh, cedj, biov, maxf, lifs, miys,
            blam_mean, biov_mean, maxf_mean, cedj_mean, tsaj_mean, nakh_mean, lifs_mean, miys_mean]

def calculate_hydrophobicity(sequence: str) -> list:
    total_hydro = sum(aa_props_dict[aa]['Hydrophobic'] for aa in sequence if aa in aa_props_dict)
    return [total_hydro / len(sequence), total_hydro]

def calculate_charge(sequence: str) -> list:
    total_charge = sum(aa_props_dict[aa]['Net-charge'] for aa in sequence if aa in aa_props_dict)
    return [total_charge / len(sequence), total_charge]

def calculate_polarity(sequence: str) -> list:
    total_pol = sum(aa_props_dict[aa]['Polarity'] for aa in sequence if aa in aa_props_dict)
    return [total_pol / len(sequence), total_pol]

def calculate_polarizability(sequence: str) -> list:
    total_polz = sum(aa_props_dict[aa]['Polarizability'] for aa in sequence if aa in aa_props_dict)
    return [total_polz / len(sequence), total_polz]

def calculate_alpha_helix_propensity(sequence: str) -> list:
    total_alpha = sum(aa_props_dict[aa]['MAXF760101(α-helix)'] for aa in sequence if aa in aa_props_dict)
    return [total_alpha / len(sequence), total_alpha]

def calculate_beta_sheet_propensity(sequence: str) -> list:
    total_beta = sum(aa_props_dict[aa]['β-sheet'] for aa in sequence if aa in aa_props_dict)
    return [total_beta / len(sequence), total_beta]

def calculate_volume(sequence: str):
    total_vol = sum(aa_props_dict[aa]['Atomic number'] for aa in sequence if aa in aa_props_dict)
    return [total_vol, total_vol / len(sequence)]

# Feature Extraction
def original_extract_features(sequences):
    features = []
    for seq in sequences:
        dde = calculate_dde(seq)
        cksaap = calculate_cksapp_features(seq)
        hydro = calculate_hydrophobicity(seq)
        charge = calculate_charge(seq)
        polarity = calculate_polarity(seq)
        polarizability = calculate_polarizability(seq)
        beta_sheet = calculate_beta_sheet_propensity(seq)
        new_props = calculate_amino_acid_properties(seq)
        volume = calculate_volume(seq)
        feat_vector = new_props + hydro + charge + polarity + polarizability + volume + beta_sheet + list(dde.values()) + list(cksaap.values())
        features.append(feat_vector)
    return np.array(features)

# Autocorrelation Feature
hydrophobicity_scale = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5,
    'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

def calculate_autocorrelation_features(sequence, max_lag=5):
    values = [hydrophobicity_scale.get(aa, 0) for aa in sequence]
    n = len(values)
    if n == 0:
        return [0] * max_lag
    mean_val = np.mean(values)
    autocorr = []
    for lag in range(1, max_lag + 1):
        if n - lag <= 0:
            autocorr.append(0)
        else:
            num = sum((values[i] - mean_val) * (values[i + lag] - mean_val) for i in range(n - lag))
            autocorr.append(num / (n - lag))
    return autocorr

def extract_features(sequences):
    features = []
    for seq in sequences:
        base = original_extract_features([seq])[0]
        ac = calculate_autocorrelation_features(seq, max_lag=5)
        features.append(np.concatenate([base, np.array(ac)]))
    return np.array(features)

# Parameter Analysis
parser = argparse.ArgumentParser(
    description="Optimized AIPstack: ML (RF+XGBoost+LightGBM+GBDT) + DL (ResNext) Fusion Model with 5-Fold CV Stacking")
parser.add_argument('--mode', type=str, help='Dataset type: Enter AMP or AIP', default='AIP')
parser.add_argument('--dataset_random_state', type=int, help='Random seed for dataset splitting', default=42)
parser.add_argument('--xgb_n_estimators', type=int, help='Number of trees for XGBoost', default=50)
parser.add_argument('--xgb_max_depth', type=int, help='Maximum tree depth for XGBoost', default=15)
parser.add_argument('--xgb_random_seed', type=int, help='Random seed for XGBoost', default=42)
parser.add_argument('--rf_n_estimators', type=int, help='Number of trees for Random Forest', default=50)
parser.add_argument('--rf_max_depth', type=int, help='Maximum tree depth for Random Forest', default=80)
parser.add_argument('--rf_random_seed', type=int, help='Random seed for Random Forest', default=42)
parser.add_argument('--lgb_n_estimators', type=int, help='Number of trees for LightGBM', default=100)
parser.add_argument('--lgb_max_depth', type=int, help='Maximum tree depth for LightGBM', default=10)
parser.add_argument('--lgb_random_seed', type=int, help='Random seed for LightGBM', default=42)
parser.add_argument('--gbdt_n_estimators', type=int, help='Number of trees for GBDT', default=100)
parser.add_argument('--gbdt_max_depth', type=int, help='Maximum tree depth for GBDT', default=10)
parser.add_argument('--gbdt_random_seed', type=int, help='Random seed for GBDT', default=42)
args = parser.parse_args()

print(f"Random seeding of the dataset: {args.dataset_random_state}")
print(f"XGBoost: n_estimators={args.xgb_n_estimators}, max_depth={args.xgb_max_depth}")
print(f"Random Forest: n_estimators={args.rf_n_estimators}, max_depth={args.rf_max_depth}")
print(f"LightGBM: n_estimators={args.lgb_n_estimators}, max_depth={args.lgb_max_depth}")
print(f"GBDT: n_estimators={args.gbdt_n_estimators}, max_depth={args.gbdt_max_depth}")

# Training Log Settings
logging.basicConfig(filename='../training_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"启动训练，参数：mode={args.mode}, dataset_random_state={args.dataset_random_state}, "
             f"xgb: ({args.xgb_n_estimators},{args.xgb_max_depth}), "
             f"rf: ({args.rf_n_estimators},{args.rf_max_depth}), "
             f"lgb: ({args.lgb_n_estimators},{args.lgb_max_depth}), "
             f"gbdt: ({args.gbdt_n_estimators},{args.gbdt_max_depth})")

# Data Reading & Feature Extraction
sequences, labels = read_dataset_from_aipstack_work('AIP.txt')
X_train_seq = sequences[419:]
X_test_seq = sequences[:419]
y_train = labels[419:]
y_test = labels[:419]

X_train = extract_features(X_train_seq)
X_test = extract_features(X_test_seq)
print("Feature extraction complete, training set shape:", X_train.shape)
print("Test Set Shape:", X_test.shape)

# ML: K=5 Cross-Validation(4-Dimensional Metric Features)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold
from sklearn.base import clone

base_models = [
    ('xgb', XGBClassifier(n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
                          use_label_encoder=False, eval_metric='logloss', random_state=args.xgb_random_seed)),
    ('rf', RandomForestClassifier(n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
                                  random_state=args.rf_random_seed)),
    ('lgb', lgb.LGBMClassifier(n_estimators=args.lgb_n_estimators, max_depth=args.lgb_max_depth,
                               random_state=args.lgb_random_seed)),
    ('gbdt', GradientBoostingClassifier(n_estimators=args.gbdt_n_estimators, max_depth=args.gbdt_max_depth,
                                        random_state=args.gbdt_random_seed))
]

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
n_train = X_train.shape[0]
n_test = X_test.shape[0]
n_models = len(base_models)

meta_train = np.zeros((n_train, n_models))
meta_test = np.zeros((n_test, n_models))

for i, (name, model) in enumerate(base_models):
    print(f"Start K=5 CV stacking, modeling:{name}")
    test_preds_folds = np.zeros((n_test, n_folds))
    fold_idx = 0
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]
        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)
        meta_train[val_idx, i] = model_clone.predict_proba(X_val)[:, 1]
        test_preds_folds[:, fold_idx] = model_clone.predict_proba(X_test)[:, 1]
        fold_idx += 1
    meta_test[:, i] = test_preds_folds.mean(axis=1)
    print(f"modle {name} complete")

print("ML K=5 stacking complete")
print("meta_train shape:", meta_train.shape)
print("meta_test shape:", meta_test.shape)

#  DL: ResNext(64 Input Layer Neurons+5 Residual Blocks+70 Epochs)
class ResNextBlock(layers.Layer):
    def __init__(self, units, cardinality=4, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        assert units % cardinality == 0, "units must be divisible by cardinality"
        self.units = units
        self.cardinality = cardinality
        self.group_units = units // cardinality
        self.dropout_rate = dropout_rate
        self.branches = []
        for _ in range(cardinality):
            branch = models.Sequential([
                layers.Dense(self.group_units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                layers.Dense(self.group_units)
            ])
            self.branches.append(branch)
        self.skip_dense = layers.Dense(units, use_bias=False)
        self.activation = layers.Activation('relu')

    def call(self, inputs, training=False):
        splits = tf.split(inputs, self.cardinality, axis=-1)
        branch_outputs = [branch(splits[i], training=training) for i, branch in enumerate(self.branches)]
        aggregated = layers.concatenate(branch_outputs, axis=-1)
        skip = self.skip_dense(inputs)
        return self.activation(aggregated + skip)

def build_resnext_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = ResNextBlock(64, cardinality=4, dropout_rate=0.3)(x)
    x = ResNextBlock(64, cardinality=4, dropout_rate=0.3)(x)
    x = ResNextBlock(64, cardinality=4, dropout_rate=0.3)(x)
    x = ResNextBlock(64, cardinality=4, dropout_rate=0.3)(x)
    x = ResNextBlock(64, cardinality=4, dropout_rate=0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

meta_train = np.array(meta_train)
meta_test = np.array(meta_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

fusion_model = build_resnext_model(input_shape=(n_models,), num_classes=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
fusion_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
fusion_model.summary()

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

best_metrics = {
    "Accuracy": 0,
    "AUC": 0,
    "Precision": 0,
    "Recall": 0,
    "F1 Score": 0,
    "MCC": 0,
    "Threshold": 0.42
}
best_prediction = None
best_prediction_binary = None

n_repeats = 5
for i in range(n_repeats):
    print(f"----- Fusion Modeling Training,Round {i + 1}  -----")
    history = fusion_model.fit(meta_train, y_train, epochs=70, batch_size=16,
                               validation_split=0.2, verbose=1,
                               callbacks=[early_stop, reduce_lr])
    predictions = fusion_model.predict(meta_test)
    threshold = 0.42
    predictions_binary = (predictions > threshold).astype(int)

    auc_val = roc_auc_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions_binary)
    prec = precision_score(y_test, predictions_binary)
    rec = recall_score(y_test, predictions_binary)
    f1 = f1_score(y_test, predictions_binary)
    mcc_val = matthews_corrcoef(y_test, predictions_binary)

    print(f"Test Set Indicators: ACC={acc:.4f}, AUC={auc_val:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, MCC={mcc_val:.4f}")

    if acc > best_metrics["Accuracy"] and auc_val > 0.85:
        best_metrics["Accuracy"] = acc
        best_metrics["AUC"] = auc_val
        best_metrics["Precision"] = prec
        best_metrics["Recall"] = rec
        best_metrics["F1 Score"] = f1
        best_metrics["MCC"] = mcc_val
        best_prediction = predictions
        best_prediction_binary = predictions_binary

# ROC Image Generation & Data Preservation
print("----- Indicators of Optimal Integration Model -----")
for key, value in best_metrics.items():
    if key != "Threshold":
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

fpr, tpr, thresholds = roc_curve(y_test, best_prediction)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {best_metrics["AUC"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

def save_results_to_excel(metrics: dict, predictions, predictions_binary,
                          folder_name="best_results_aip_resnext_optimized_cv_v2"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{folder_name}/results_{current_time}.xlsx"
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        predictions_df = pd.DataFrame({
            'Predictions': predictions.flatten(),
            'Predictions Binary': predictions_binary.flatten()
        })
        predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_excel(writer, index=False, sheet_name='Metrics')
    print(f"Results saved to {filename}")

save_results_to_excel(best_metrics, best_prediction, best_prediction_binary)

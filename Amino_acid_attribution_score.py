import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib import transforms
import matplotlib.colors as mcolors  
import collections

# -----------------------------
# Global settings: use English fonts
# -----------------------------
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Predefined color mapping for 20 standard amino acids
# -----------------------------
amino_acid_colors = {
    'A': '#e6194b',
    'C': '#3cb44b',
    'D': '#ffe119',
    'E': '#0082c8',
    'F': '#f58231',
    'G': '#911eb4',
    'H': '#46f0f0',
    'I': '#d2f53c',  
    'K': '#f032e6',  
    'L': '#fabebe',
    'M': '#008080',
    'N': '#e6beff',
    'P': '#aa6e28',
    'Q': '#fffac8',
    'R': '#800000',
    'S': '#aaffc3',
    'T': '#808000',
    'V': '#ffd8b1',
    'W': '#808080',
    'Y': '#000080',
}
standard_alphabet = list(amino_acid_colors.keys())

# -----------------------------
# Define num_layers globally
# -----------------------------
num_layers = 5

# -----------------------------
# Custom gradient colors (for heatmaps)
# -----------------------------
# Positive (AIP) 
cmap_positive = mcolors.LinearSegmentedColormap.from_list("positive_cmap", ["#FFFFFF", "#EA6C84"])
# Negative (Non-AIP) 
cmap_negative = mcolors.LinearSegmentedColormap.from_list("negative_cmap", ["#FFFFFF", "#5AA3CB"])

# -----------------------------
# 1. Model Attribution Scores based on biochemical properties
# -----------------------------
def model_attribution_scores(seq, num_layers=5):
    """
    For a given sequence seq, return a matrix with shape=(num_layers, len(seq)) by the following properties:
      Layer 1: Hydrophobicity (Kyte–Doolittle scale)
      Layer 2: Net Charge (K, R: +1; D, E: -1; others: 0)
      Layer 3: Polarity (approximate)
      Layer 4: Molecular Weight (Dalton)
      Layer 5: Side Chain Volume (approximate)
    Undefined residues are assigned a value of 0.
    """
    hydro = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
             'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
             'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
             'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
    charge = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
              'G': 0, 'H': 0, 'I': 0, 'K': 1, 'L': 0,
              'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
              'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
    polarity = {'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.3, 'F': 5.2,
                'G': 9.0, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9,
                'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
                'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2}
    mw = {'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
          'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
          'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
          'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181}
    vol = {'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
           'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
           'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
           'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6}
    properties = [hydro, charge, polarity, mw, vol]
    if num_layers < 5:
        properties = properties[:num_layers]
    elif num_layers > 5:
        properties += [vol]*(num_layers-5)
    mat = np.zeros((num_layers, len(seq)))
    for i, aa in enumerate(seq):
        for j in range(num_layers):
            mat[j, i] = properties[j].get(aa, 0)
    return mat

# -----------------------------
# 2. Read sequences from file and group them
# -----------------------------
def read_sequences(file_path):
    """
    Read the file in groups of two rows:
    First row of tags (1=AIP, 0=Non-AIP)
    Second row is the amino acid sequence
    """
    labels = []
    sequences = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    for i in range(0, len(lines), 2):
        labels.append(int(lines[i]))
        sequences.append(lines[i+1])
    return labels, sequences

# Please change the filename to the actual document
labels, seqs = read_sequences('AIP.txt')
aip_seqs = [s for l, s in zip(labels, seqs) if l == 1]
non_aip_seqs = [s for l, s in zip(labels, seqs) if l == 0]

# -----------------------------
# 3. Aggregate Group Attribution and perform per-layer normalization
# -----------------------------
def aggregate_group_attribution(group_samples, num_layers, alphabet=standard_alphabet):
    """
    For each sample of group_samples, an imputation matrix was obtained using model_attribution_scores.
    For each stratum, imputation scores and counts were accumulated for the standard amino acid pair
    The mean value was calculated. Finally, min-max normalization is applied to each layer so that the values in each layer are mapped to [0,1].
    Return matrix shape=(num_layers, len(alphabet))
    """
    sum_scores = {layer: {aa: 0.0 for aa in alphabet} for layer in range(num_layers)}
    count_scores = {layer: {aa: 0 for aa in alphabet} for layer in range(num_layers)}
    for seq in group_samples:
        scores = model_attribution_scores(seq, num_layers)
        for pos, aa in enumerate(seq):
            if aa in alphabet:
                for layer in range(num_layers):
                    sum_scores[layer][aa] += scores[layer, pos]
                    count_scores[layer][aa] += 1
    matrix = np.zeros((num_layers, len(alphabet)))
    for layer in range(num_layers):
        for j, aa in enumerate(alphabet):
            if count_scores[layer][aa] > 0:
                matrix[layer, j] = sum_scores[layer][aa] / count_scores[layer][aa]
            else:
                matrix[layer, j] = 0
        # 对每层进行 min-max 归一化
        row_min = matrix[layer].min()
        row_max = matrix[layer].max()
        if row_max > row_min:
            matrix[layer] = (matrix[layer] - row_min) / (row_max - row_min)
        else:
            matrix[layer] = 0
    return matrix, alphabet

agg_aip_matrix, agg_order = aggregate_group_attribution(aip_seqs, num_layers, standard_alphabet)
agg_non_aip_matrix, _ = aggregate_group_attribution(non_aip_seqs, num_layers, standard_alphabet)

# -----------------------------
# 4. Draw Aggregated Heatmap (pass custom cmap parameter)
# -----------------------------
def plot_aggregated_heatmap(matrix, residue_order, title, cmap='viridis'):
    plt.figure(figsize=(max(8, len(residue_order) * 0.5), 4))
    im = plt.imshow(matrix, aspect='auto', interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold', color='black')
    plt.xlabel("Residue", fontsize=12, fontweight='bold', color='black')
    plt.ylabel("Layer", fontsize=12, fontweight='bold', color='black')
    plt.xticks(ticks=np.arange(len(residue_order)), labels=residue_order, rotation=90, fontsize=10)
    plt.yticks(ticks=np.arange(matrix.shape[0]), labels=[f"Layer {i+1}" for i in range(matrix.shape[0])], fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), fontweight='bold', color='black')
    plt.setp(plt.gca().get_yticklabels(), fontweight='bold', color='black')
    # Generate a colorbar and bold the scale labels in it.
    cbar = plt.colorbar(im)
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold', color='black')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 5. Get Order of Amino Acids as they first appear (for logo and bar charts)
# -----------------------------
def get_order(seq_list, alphabet=standard_alphabet):
    order = []
    for seq in seq_list:
        for ch in seq:
            if ch in alphabet and ch not in order:
                order.append(ch)
    return order

global_order = get_order(seqs, standard_alphabet)         # Global order for bar charts
order_aip = get_order(aip_seqs, standard_alphabet)          # AIP logo sequences
order_non_aip = get_order(non_aip_seqs, standard_alphabet)    # Non-AIP logo sequences

# -----------------------------
# 6. Compute Amino Acid Count (Raw Count)
# -----------------------------
def compute_amino_acid_count(seq_list, order):
    count_dict = {aa: 0 for aa in order}
    for seq in seq_list:
        for ch in seq:
            if ch in count_dict:
                count_dict[ch] += 1
    return count_dict

count_aip_ordered = compute_amino_acid_count(aip_seqs, order_aip)
count_non_aip_ordered = compute_amino_acid_count(non_aip_seqs, order_non_aip)
count_aip_global = compute_amino_acid_count(aip_seqs, global_order)
count_non_aip_global = compute_amino_acid_count(non_aip_seqs, global_order)

# -----------------------------
# 7. Plot Motif-like Logo (without logomaker)
# -----------------------------
def plot_amino_acid_logo(count_dict, order, title, color_map=amino_acid_colors, gap=0.1):
    """
    Draw a logo diagram that resembles a DNA motif:
      - The horizontal axis lists the amino acids in order;
      - The height of the letters is normalized by the number of occurrences (maximum height is set to 5);
      - To avoid overlap, each letter is followed by a fixed-pitch gap;
      - The title, axis labels, and scale text are bolded in black;
      - Amino acid graphs are in the original set of colors.
    """
    max_count = max(count_dict.values()) if count_dict else 1e-9
    if max_count <= 0:
        max_count = 1e-9
    fig, ax = plt.subplots(figsize=(max(8, len(order)), 4))
    x_pos = 0
    for aa in order:
        cnt = count_dict[aa]
        letter_height = cnt / max_count * 5   # Maximum letter height of 5 units
        tp = TextPath((0, 0), aa, size=1,
                      prop=matplotlib.font_manager.FontProperties(family='Arial'))
        bb = tp.get_extents()
        base_width = bb.width
        base_height = bb.height
        sx = 1.0 / base_width  # Fixed letter width of 1 unit
        sy = letter_height / base_height
        transform = transforms.Affine2D().scale(sx, sy).translate(x_pos, 0)
        path = transform.transform_path(tp)
        patch = PathPatch(path, color=color_map.get(aa, 'black'), lw=0)
        ax.add_patch(patch)
        x_pos += 1 + gap
    ax.set_xlim(-0.5, x_pos - gap + 0.5)
    ax.set_ylim(0, 6)
    ax.set_xticks(np.arange(0, len(order) * (1 + gap), (1 + gap)))
    ax.set_xticklabels(order, rotation=90)
    plt.setp(ax.get_xticklabels(), fontweight='bold', color='black')
    plt.setp(ax.get_yticklabels(), fontweight='bold', color='black')
    ax.set_ylabel("Normalized Count (relative scale)", fontsize=12, fontweight='bold', color='black')
    ax.set_title(title, fontsize=14, fontweight='bold', color='black')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 8. Plot Grouped Bar Chart (Count Comparison)
# -----------------------------
def plot_count_bar_chart(count_dict_aip, count_dict_non_aip, order, title):
    x = np.arange(len(order))
    counts_a = [count_dict_aip.get(aa, 0) for aa in order]
    counts_n = [count_dict_non_aip.get(aa, 0) for aa in order]
    width = 0.4
    plt.figure(figsize=(max(8, len(order) * 0.4), 4))
    plt.bar(
        x - width / 2, counts_a, width,
        color='#EA6C84', edgecolor='black', label='AIP', alpha=0.8
    )
    plt.bar(
        x + width / 2, counts_n, width,
        color='#5AA3CB', edgecolor='black', label='Non-AIP', alpha=0.8
    )
    plt.xticks(x, order, rotation=90, fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), fontweight='bold', color='black')
    # Set the y-axis scale to black and bold
    plt.setp(plt.gca().get_yticklabels(), fontweight='bold', color='black')
    plt.xlabel("Amino Acid", fontsize=12, fontweight='bold', color='black')
    plt.ylabel("Count", fontsize=12, fontweight='bold', color='black')
    plt.title(title, fontsize=14, fontweight='bold', color='black')
    plt.legend(prop={'weight': 'bold'}, edgecolor='black')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 9. Generate Final 5 Plots
# -----------------------------

# (1) AIP Aggregated Attribution Distribution Heatmap(using Positive gradient colors)
plot_aggregated_heatmap(
    agg_aip_matrix,
    standard_alphabet,
    "AIP Aggregated Attribution Distribution Heatmap",
    cmap=cmap_positive
)

# (2) Non-AIP Aggregated Attribution Distribution Heatmap(using Negative gradient colors)
plot_aggregated_heatmap(
    agg_non_aip_matrix,
    standard_alphabet,
    "Non-AIP Aggregated Attribution Distribution Heatmap",
    cmap=cmap_negative
)

# (3) AIP Motif-like Amino Acid Logo
plot_amino_acid_logo(count_aip_ordered, order_aip, "AIP Motif-like Amino Acid Logo")

# (4) Non-AIP Motif-like Amino Acid Logo
plot_amino_acid_logo(count_non_aip_ordered, order_non_aip, "Non-AIP Motif-like Amino Acid Logo")

# (5) Amino Acid Count Comparison Bar Chart (using global order)
plot_count_bar_chart(count_aip_global, count_non_aip_global, global_order,
                     "Amino Acid Count Comparison (AIP vs Non-AIP)")

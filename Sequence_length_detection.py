import matplotlib.pyplot as plt
import numpy as np

# 1) Read the file
with open("AIP.txt", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

positive_lengths = []
negative_lengths = []

# 2) Parse labels and sequences
for i in range(0, len(lines), 2):
    if i + 1 < len(lines):
        label = lines[i].strip()
        sequence = lines[i + 1].strip()
        length = len(sequence)
        if label == "1":
            positive_lengths.append(length)
        elif label == "0":
            negative_lengths.append(length)

# 3) Print counts of positive, negative, and total samples
num_positive = len(positive_lengths)
num_negative = len(negative_lengths)
total_samples = num_positive + num_negative

print("Number of positive samples:", num_positive)
print("Number of negative samples:", num_negative)
print("Total number of anti-inflammatory peptides:", total_samples)

# 4) Define bins from 10 to 35 with a step of 5
#    This creates bin edges: [10, 15, 20, 25, 30, 35]
bins = np.arange(10, 40, 5)
# Create labels like "10-15AA", "15-20AA", etc.
bin_labels = [f"{bins[i]}-{bins[i+1]}AA" for i in range(len(bins) - 1)]

# 5) Count how many lengths fall into each bin for positive and negative samples
pos_counts, _ = np.histogram(positive_lengths, bins=bins)
neg_counts, _ = np.histogram(negative_lengths, bins=bins)

# Prepare y positions
y_positions = np.arange(len(bin_labels))

# Identify the maximum count in positive and negative (the "longest" bars)
max_pos_count = np.max(pos_counts) if len(pos_counts) > 0 else 0
max_neg_count = np.max(neg_counts) if len(neg_counts) > 0 else 0

# 设置图像大小，宽度保持8，高度调整为4
plt.figure(figsize=(8, 4))

# Plot positive bars to the right
plt.barh(y_positions, pos_counts, color='salmon', label='Positive', edgecolor='black')
# Plot negative bars to the left (by using negative values)
plt.barh(y_positions, -neg_counts, color='lightblue', label='Negative', edgecolor='black')

# Choose an offset so that smaller bars' labels appear a bit away from the bar
OFFSET = 100

# Annotate positive bars
for i, count in enumerate(pos_counts):
    if count <= 0:
        continue
    if count == max_pos_count:
        # Place the text in the middle of the bar
        plt.text(count / 2, y_positions[i], str(count),
                 va='center', ha='center', fontsize=10, fontweight='bold', color='black')
    else:
        # Place the text to the right of the bar
        plt.text(count + OFFSET, y_positions[i], str(count),
                 va='center', ha='left', fontsize=10, fontweight='bold', color='black')

# Annotate negative bars
for i, count in enumerate(neg_counts):
    if count <= 0:
        continue
    if count == max_neg_count:
        # Place the text in the middle of the bar
        plt.text(-count / 2, y_positions[i], str(count),
                 va='center', ha='center', fontsize=10, fontweight='bold', color='black')
    else:
        # Place the text to the left of the bar
        plt.text(-count - OFFSET, y_positions[i], str(count),
                 va='center', ha='right', fontsize=10, fontweight='bold', color='black')

# Draw a vertical line at x=0 to separate negative and positive sides
plt.axvline(0, color='black', linewidth=1)

# Set y-ticks with our bin labels
plt.yticks(y_positions, bin_labels, fontsize=10, fontweight='bold', color='black')
# 同时设置 x 轴刻度标签样式（若有需要）
plt.xticks(fontsize=10, fontweight='bold', color='black')

# Axis labels and title in English (均为黑色加粗)
plt.xlabel("Number of Peptides", fontsize=12, fontweight='bold', color='black')
plt.title("AIP Sequence Length Distribution", fontsize=14, fontweight='bold', color='black')

# Legend设置字体加粗
plt.legend(prop={'weight': 'bold'}, edgecolor='black')

plt.tight_layout()
plt.show()

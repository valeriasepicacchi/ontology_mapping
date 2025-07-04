import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from scipy.interpolate import griddata
import random
import re
from mpl_toolkits.mplot3d import Axes3D

def threed_barplot(data):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Data
    metrics = ['Precision', 'Recall', 'F1-score', 'Support']
    features = ['First', 'Second', 'Third', 'Fourth', 'Fifth']


    _x = np.arange(len(metrics))
    _y = np.arange(len(features))
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    z = np.zeros_like(x)

    dx = dy = 0.3
    dz = data.ravel()

    ax.bar3d(x, y, z, dx, dy, dz, color=plt.cm.viridis(dz / dz.max()))

    # Labels and ticks
    ax.set_xlabel('Metric')
    ax.set_ylabel('Feature Group')
    ax.set_zlabel('Score')

    ax.set_xticks(_x + 0.15)
    ax.set_xticklabels(metrics, rotation=45, ha='right')

    ax.set_yticks(_y + 0.15)
    ax.set_yticklabels(features)

    plt.title('3D Bar Plot of Metrics for Each Feature Group', fontsize=14)
    plt.tight_layout()
    plt.show()

def surface_plot(data):

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    metrics = ['Precision', 'Recall', 'F1-score', 'Support']
    features = ['First', 'Second', 'Third', 'Fourth', 'Fifth']
    # Prepare mesh grid
    x = np.arange(len(metrics))
    y = np.arange(len(features))
    x, y = np.meshgrid(x, y)
    z = data

    # Plot surface
    surf = ax.plot_surface(x, y, z, cmap='plasma', edgecolor='none', alpha=0.8)

    # Labels
    ax.set_xlabel('Metric')
    ax.set_ylabel('Feature Group')
    ax.set_zlabel('Score')

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')

    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.title('3D Surface Plot of Feature Scores', fontsize=14)
    plt.tight_layout()
    plt.show()

def three_scatter(data):


    metrics = ['Precision', 'Recall', 'F1-score']
    features = ['First', 'Second', 'Third', 'Fourth', 'Fifth']


    # Extract the main metric values and support as a separate array
    metric_values = data[:, :-1]
    supports = data[:, -1]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting bubbles for each feature
    for i, feature in enumerate(features):
        xs = np.arange(len(metrics))
        ys = [i] * len(metrics)
        zs = metric_values[i]
        bubble_sizes = supports[i] * 100  # scale to visually meaningful sizes

        ax.scatter(xs, ys, zs, s=bubble_sizes, alpha=0.7, label=feature)

    # Labels and ticks
    ax.set_xlabel('Metric')
    ax.set_ylabel('Feature Group')
    ax.set_zlabel('Score')

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)

    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)

    ax.set_title('3D Bubble Plot: Feature Metrics with Support as Bubble Size', fontsize=14)
    ax.legend(title='Feature', loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()



def bubble_plot(data):

    thresholds = [0.5, 0.7, 0.8, 0.9, 'cos>0.5+max_neg=2']  
    metrics = ['Precision', 'Recall', 'F1-score']

    # Convert to arrays
    data = np.array(data)
    metric_values = data[:, :3]
    supports = data[:, 3]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Loop over thresholds
    for i, (threshold_label, values, support) in enumerate(zip(thresholds, metric_values, supports)):
        xs = [i] * len(metrics) 
        ys = np.arange(len(metrics))  
        zs = values  # Metric values

        ax.scatter(xs, ys, zs, s=support/2, alpha=0.7, label=str(threshold_label), edgecolor='k')

    ax.set_xlabel('Threshold Index')
    ax.set_ylabel('Metric Type')
    ax.set_zlabel('Metric Value')
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)

    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds], rotation=20)

    ax.set_title('Metrics vs Threshold with Support as Bubble Size', fontsize=14)
    ax.legend(title='Thresholds', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()



def read_latex_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def parse_latex_results_structured(latex_text):
    tables = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', latex_text, re.DOTALL)

    all_tables = []

    for table in tables:
        # Extract threshold label
        threshold_match = re.search(r'\\multicolumn\{5\}\{\|c\|\}\{(.*?)\}', table)
        if threshold_match:
            threshold_label = threshold_match.group(1).strip()
        else:
            threshold_label = 'Unknown'

        # Extract individual parameters
        max_neg_match = re.search(r'max negatives\s*=\s*(\d+)', threshold_label)
        fuzz_partial_match = re.search(r'fuzz partial ratio\s*>\s*([\d\.]+)', threshold_label)
        cosine_sim_match = re.search(r'cosine similarity\s*>\s*([\d\.]+)', threshold_label)

        max_negatives = int(max_neg_match.group(1)) if max_neg_match else None
        fuzz_partial_ratio = float(fuzz_partial_match.group(1)) if fuzz_partial_match else None
        cosine_similarity = float(cosine_sim_match.group(1)) if cosine_sim_match else None

        # Split rows
        rows = re.split(r'\\\\\s*(?:\\hline)?', table)
        parsed_rows = []

        for row in rows:
            row = row.strip()
            if not row or row.startswith('&') or row.startswith('\\'):
                continue

            cols = [col.strip() for col in row.split('&')]
            if len(cols) != 5:
                continue

            label = cols[0]
            try:
                precision = float(cols[1]) if cols[1] else None
                recall = float(cols[2]) if cols[2] else None
                f1_score = float(cols[3]) if cols[3] else None
                support = float(cols[4]) if cols[4] else None
            except ValueError:
                continue

            parsed_rows.append({
                'label': label,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support
            })

        # Save the full table structure
        all_tables.append({
            'threshold_label': threshold_label,
            'max_negatives': max_negatives,
            'fuzz_partial_ratio': fuzz_partial_ratio,
            'cosine_similarity': cosine_similarity,
            'rows': parsed_rows
        })

    return all_tables

def support_bubble_plot(data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Flatten out arrays for plotting
    xs = []
    ys = []
    zs = []
    supports_flat = []
    thresholds = [0.5, 0.7, 0.8, 0.9, 'cos>0.5+max_neg=2']  
    metrics = ['Precision', 'Recall', 'F1-score']
    data = np.array(data)
    metric_values = data[:, :3]
    supports = data[:, 3]

    for i, (threshold_label, values, support) in enumerate(zip(thresholds, metric_values, supports)):
        for j, value in enumerate(values):
            xs.append(i)
            ys.append(j)
            zs.append(value)
            supports_flat.append(support)

    # Plot with color mapping support
    sc = ax.scatter(xs, ys, zs, s=np.array(supports_flat)/2, c=supports_flat, cmap='viridis', alpha=0.7, edgecolor='k')

    # Colorbar to show support scale
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Support', rotation=270, labelpad=15)

    # Labels
    ax.set_xlabel('Threshold Index')
    ax.set_ylabel('Metric Type')
    ax.set_zlabel('Metric Value')
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds], rotation=20)
    ax.set_title('Metrics vs Thresholds (Support as Bubble Size & Color)', fontsize=14)

    plt.tight_layout()
    plt.show()


    ''' macroaverage'''

def bubble_3d_plot(dataset_name,all_tables, metric_label='macro-avg', metric_field='f1_score'):
    xs = []
    ys = []
    zs = []
    sizes = []

    for table in all_tables:
        cosine = table['cosine_similarity']
        fuzz = table['fuzz_partial_ratio']
        
        # Trova riga macro-avg
        try:
            metric_row = next(row for row in table['rows'] if row['label'] == metric_label)
            metric_value = metric_row[metric_field]
            support = metric_row['support']
        except StopIteration:
            continue  
        
        # Save data
        xs.append(cosine)
        ys.append(fuzz)
        zs.append(metric_value)
        sizes.append(support)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xs, ys, zs, s=np.array(sizes)/5, c=zs, cmap='plasma', alpha=0.8, edgecolor='k')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(f'{metric_field}', rotation=270, labelpad=15)

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Fuzz Partial Ratio')
    ax.set_zlabel(metric_field.capitalize())
    ax.set_title(f'3D Bubble Plot: {metric_label} - {metric_field}', fontsize=14)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/plots/bubble_3d_plot_{dataset_name}.png", dpi=300)

def scatter_3d_plot(dataset_name,all_tables, metric_label='macro-avg', metric_field='f1_score'):
    xs = []
    ys = []
    zs = []

    for table in all_tables:
        cosine = table['cosine_similarity']
        fuzz = table['fuzz_partial_ratio']
        
        # Trova riga macro-avg
        try:
            metric_row = next(row for row in table['rows'] if row['label'] == metric_label)
            metric_value = metric_row[metric_field]
        except StopIteration:
            continue  # se macro-avg non c'è, salta
        
        # Save data
        xs.append(cosine)
        ys.append(fuzz)
        zs.append(metric_value)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xs, ys, zs, s=50, c=zs, cmap='plasma', alpha=0.9, edgecolor='k')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(f'{metric_field}', rotation=270, labelpad=15)

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Fuzz Partial Ratio')
    ax.set_zlabel(metric_field.capitalize())
    ax.set_title(f'3D Scatter Plot: {metric_label} - {metric_field}', fontsize=14)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/plots/scatter_3d_plot_{dataset_name}.png", dpi=300)

def surface_3d_plot(dataset_name,all_tables, metric_label='macro-avg', metric_field='f1_score'):

    xs = []
    ys = []
    zs = []

    for table in all_tables:
        cosine = table['cosine_similarity']
        fuzz = table['fuzz_partial_ratio']
        
        try:
            metric_row = next(row for row in table['rows'] if row['label'] == metric_label)
            metric_value = metric_row[metric_field]
        except StopIteration:
            continue
        
        xs.append(cosine)
        ys.append(fuzz)
        zs.append(metric_value)

    # Create grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(xs), max(xs), 50),
        np.linspace(min(ys), max(ys), 50)
    )

    # Interpolate Z values on grid
    grid_z = griddata((xs, ys), zs, (grid_x, grid_y), method='cubic')

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.8)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(f'{metric_field}', rotation=270, labelpad=15)

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Fuzz Partial Ratio')
    ax.set_zlabel(metric_field.capitalize())
    ax.set_title(f'3D Surface Plot: {metric_label} - {metric_field}', fontsize=14)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/plots/surface_3d_plot_{dataset_name}.png", dpi=300)


def bubble_3d_plot_with_support(dataset_name,all_tables, metric_label='macro-avg', metric_field='f1_score'):

    xs = []
    ys = []
    zs = []
    supports = []

    for table in all_tables:
        cosine = table['cosine_similarity']
        fuzz = table['fuzz_partial_ratio']
        
        try:
            metric_row = next(row for row in table['rows'] if row['label'] == metric_label)
            metric_value = metric_row[metric_field]
            support_value = metric_row['support']
        except StopIteration:
            continue
        
        xs.append(cosine)
        ys.append(fuzz)
        zs.append(metric_value)
        supports.append(support_value)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize support for bubble size
    supports_size = np.array(supports) / np.max(supports) * 300  # size scaling factor
    sc = ax.scatter(xs, ys, zs, s=supports_size, c=supports, cmap='viridis', alpha=0.8, edgecolor='k')

    # Colorbar to show support scale
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Support', rotation=270, labelpad=15)

    # Labels
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Fuzz Partial Ratio')
    ax.set_zlabel(metric_field.capitalize())
    ax.set_title(f'3D Bubble Plot: {metric_label} - {metric_field} (size & color = Support)', fontsize=14)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/plots/bubble_3d_plot_with_support_{dataset_name}.png", dpi=300)

def barplot_last5_with_support(dataset_name,all_tables, metric_label='macro-avg'):
    # Prendi ultime 5 tabelle
    last_tables = all_tables[-5:]

    thresholds_labels = []
    precision_values = []
    recall_values = []
    f1_values = []
    support_values = []

    for table in last_tables:
        cosine = table['cosine_similarity']
        fuzz = table['fuzz_partial_ratio']
        label = f'cos>{cosine:.2f}\nfuzz>{fuzz}'
        thresholds_labels.append(label)

        try:
            metric_row = next(row for row in table['rows'] if row['label'] == metric_label)
            precision_values.append(metric_row['precision'])
            recall_values.append(metric_row['recall'])
            f1_values.append(metric_row['f1_score'])
            support_values.append(metric_row['support'])
        except StopIteration:
            precision_values.append(0)
            recall_values.append(0)
            f1_values.append(0)
            support_values.append(0)

    # Normalizza support tra 0 e 1 per confrontarlo con le altre metriche
    support_values_norm = np.array(support_values) / max(support_values)

    # Barplot
    x = np.arange(len(thresholds_labels))  # posizione gruppi
    width = 0.2  # larghezza barre

    fig, ax = plt.subplots(figsize=(14, 6))

    rects1 = ax.bar(x - 1.5*width, precision_values, width, label='Precision')
    rects2 = ax.bar(x - 0.5*width, recall_values, width, label='Recall')
    rects3 = ax.bar(x + 0.5*width, f1_values, width, label='F1-score')
    rects4 = ax.bar(x + 1.5*width, support_values_norm, width, label='Support (normalized)')

    # Labels
    ax.set_xlabel('Threshold (cosine similarity & fuzz partial ratio)')
    ax.set_ylabel('Value')
    ax.set_title(f'{metric_label} - Comparison of Metrics + Support (Last 5 thresholds)')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds_labels)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Annotazioni sulle barre (opzionale ma utile per tesi!)
    def autolabel(rects, values):
        for rect, val in zip(rects, values):
            height = rect.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # offset sopra la barra
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1, precision_values)
    autolabel(rects2, recall_values)
    autolabel(rects3, f1_values)
    autolabel(rects4, support_values_norm)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/plots/barplot_last5_with_support_{dataset_name}.png", dpi=300)

def barplot_randomN_sorted_with_support(dataset_name,all_tables, N=8, metric_label='macro-avg', random_seed=42):
    # Set seed for reproducibility
    random.seed(random_seed)
    selected_tables = random.sample(all_tables, N)
    selected_tables_sorted = sorted(selected_tables, key=lambda table: table['cosine_similarity'])

    thresholds_labels = []
    precision_values = []
    recall_values = []
    f1_values = []
    support_values = []

    for table in selected_tables_sorted:
        cosine = table['cosine_similarity']
        fuzz = table['fuzz_partial_ratio']
        label = f'cos>{cosine:.2f}\nfuzz>{fuzz}'
        thresholds_labels.append(label)

        try:
            metric_row = next(row for row in table['rows'] if row['label'] == metric_label)
            precision_values.append(metric_row['precision'])
            recall_values.append(metric_row['recall'])
            f1_values.append(metric_row['f1_score'])
            support_values.append(metric_row['support'])
        except StopIteration:
            precision_values.append(0)
            recall_values.append(0)
            f1_values.append(0)
            support_values.append(0)

    # Normalizza support tra 0 e 1 per confrontarlo con le altre metriche
    support_values_norm = np.array(support_values) / max(support_values)

    # Barplot
    x = np.arange(len(thresholds_labels))  # posizione gruppi
    width = 0.2  # larghezza barre

    fig, ax = plt.subplots(figsize=(16, 6))

    rects1 = ax.bar(x - 1.5*width, precision_values, width, label='Precision')
    rects2 = ax.bar(x - 0.5*width, recall_values, width, label='Recall')
    rects3 = ax.bar(x + 0.5*width, f1_values, width, label='F1-score')
    rects4 = ax.bar(x + 1.5*width, support_values_norm, width, label='Support (normalized)')

    # Labels
    ax.set_xlabel('Threshold (cosine similarity & fuzz partial ratio)')
    ax.set_ylabel('Value')
    ax.set_title(f'{metric_label} - Comparison of Metrics + Support (Random {N} thresholds, Sorted)')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds_labels)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Annotazioni sulle barre
    def autolabel(rects, values):
        for rect, val in zip(rects, values):
            height = rect.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # offset sopra la barra
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1, precision_values)
    autolabel(rects2, recall_values)
    autolabel(rects3, f1_values)
    autolabel(rects4, support_values_norm)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/plots/barplot_randomN_sorted_with_support_{dataset_name}.png", dpi=300)

def barplot_randomN_sorted_dualaxis(dataset_name,all_tables, N=8, metric_label='macro-avg', random_seed=42):
    random.seed(random_seed)
    selected_tables = random.sample(all_tables, N)
    selected_tables_sorted = sorted(selected_tables, key=lambda table: table['cosine_similarity'])

    thresholds_labels = []
    precision_values = []
    recall_values = []
    f1_values = []
    support_values = []

    for table in selected_tables_sorted:
        cosine = table['cosine_similarity']
        fuzz = table['fuzz_partial_ratio']
        label = f'cos>{cosine:.2f}\nfuzz>{fuzz}'
        thresholds_labels.append(label)

        try:
            metric_row = next(row for row in table['rows'] if row['label'] == metric_label)
            precision_values.append(metric_row['precision'])
            recall_values.append(metric_row['recall'])
            f1_values.append(metric_row['f1_score'])
            support_values.append(metric_row['support'])
        except StopIteration:
            precision_values.append(0)
            recall_values.append(0)
            f1_values.append(0)
            support_values.append(0)

    x = np.arange(len(thresholds_labels))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Asse principale (metriche)
    rects1 = ax1.bar(x - width, precision_values, width, label='Precision', color='tab:blue')
    rects2 = ax1.bar(x, recall_values, width, label='Recall', color='tab:orange')
    rects3 = ax1.bar(x + width, f1_values, width, label='F1-score', color='tab:green')

    ax1.set_xlabel('Threshold (cosine similarity & fuzz partial ratio)')
    ax1.set_ylabel('Metric Value')
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(thresholds_labels)
    ax1.legend(loc='upper left')

    # Asse secondario (support)
    ax2 = ax1.twinx()
    ax2.plot(x, support_values, 'o-', color='tab:red', linewidth=2, label='Support')
    ax2.set_ylabel('Support (number of instances)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Aggiungi legenda per Support
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc='upper center')

    plt.title(f'{metric_label} - Metrics + Support (Random {N} thresholds, Sorted)')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/plots/barplot_randomN_sorted_dualaxis_{dataset_name}.png", dpi=300)
# threed_barplot(data)
# surface_plot(data)
# three_scatter(data)
# support_bubble_plot()


# Example data: thresholds (as "x"), metrics (as "y"), scores (as "z")
thresholds = np.array([0.5, 0.7, 0.8, 0.9])
metrics = np.array([0, 1, 2])  # 0: precision, 1: recall, 2: f1-score

latex_text = read_latex_file('/Users/sepicacchiv/Desktop/thesis/results/classification_report_latex.txt') 

# Parse LaTeX table
all_tables = parse_latex_results_structured(latex_text)

# Esempio: print prima tabella
bubble_3d_plot('ncit_doid_draft_',all_tables, metric_label='1')
scatter_3d_plot('ncit_doid_draft_',all_tables)
surface_3d_plot('ncit_doid_draft_',all_tables)
bubble_3d_plot_with_support('ncit_doid_draft_',all_tables)
barplot_last5_with_support('ncit_doid_draft_',all_tables)
barplot_randomN_sorted_with_support('ncit_doid_draft_',all_tables)
barplot_randomN_sorted_dualaxis('ncit_doid_draft_',all_tables)

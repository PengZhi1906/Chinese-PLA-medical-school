import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np

def plot_gene_expression(file_path):
    # Load data from TXT file (assuming tab-separated values)
    df = pd.read_csv(file_path, delimiter='\t')

    # Applying logarithmic transformation to gene expression values
    genes = df.columns[1:-1]
    for gene in genes:
        # Apply log transformation, adding a small value to avoid log(0)
        df[gene] = np.log(df[gene] + 1)

    # Melting the DataFrame to long format for easier plotting
    df_long = pd.melt(df, id_vars=["ID", "risk"], value_vars=genes,
                      var_name="Gene", value_name="Expression")

    # Creating a boxplot with different colors for each risk group
    plt.figure(figsize=(12, 6))
    boxplot = sns.boxplot(x="Gene", y="Expression", hue="risk", data=df_long, palette="Set1", width=0.6)

    # Adjusting y-axis range
    boxplot.set_ylim(0, 15)

    # Moving the legend to a better position
    plt.legend(loc='upper left')

    # Calculating and annotating p-values, adjusting their positions
    for i, gene in enumerate(genes):
        high = df[df['risk'] == 'High risk'][gene]
        low = df[df['risk'] == 'Low risk'][gene]
        t_stat, p_val = ttest_ind(high, low)
        plt.text(i, 12, f'p={p_val:.2e}',  # Adjusted to be near the top of the plot
                 horizontalalignment='center', size='medium', color='black', weight='semibold')


    plt.show()


# Example usage
file_path = 'F:\\博士期间\\图神经网络\\预后标志物\\8.boxplot/expSite.txt'  # Replace with your file path
plot_gene_expression(file_path)
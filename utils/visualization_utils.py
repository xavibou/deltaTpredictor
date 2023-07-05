import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_results(predicted, ground_truth):
    
    plt.figure()
    # Plotting predicted vs ground truth
    plt.scatter(ground_truth, predicted)
    plt.ylabel('Predicted')
    plt.xlabel('Ground Truth')
    plt.title('Predicted vs Ground Truth')
    plt.ylim([0, np.max(ground_truth)])
    plt.show()
    plt.savefig('validation_correlation.png')

    # Computing correlation
    correlation, p_value = pearsonr(predicted, ground_truth)
    print(f"Correlation: {correlation:.2f}")
    print(f"P-value: {p_value:.2f}")

   
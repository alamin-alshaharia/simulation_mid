import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def generate_unimodal_sample(mean, std, sample_size):
    return np.random.normal(mean, std, sample_size)
def generate_multimodal_sample(mean1, mean2, std, sample_size):

    sample1 = np.random.normal(mean1, std, sample_size // 2)
    sample2 = np.random.normal(mean2, std, sample_size // 2)
    return np.concatenate([sample1, sample2])
def plot_distributions(unimodal_sample, multimodal_sample, diastolic_bp_sample):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(unimodal_sample, kde=True, color='blue', stat='density', bins=20)
    plt.title('Unimodal Normal Distribution\nMean=100, SD=20')
    plt.xlabel('Value')
    plt.ylabel('Density')

    plt.subplot(2, 2, 2)
    sns.histplot(multimodal_sample, kde=True, color='green', stat='density', bins=20)
    plt.title('Multimodal Normal Distribution\nMeans=100 & 130, SD=20')
    plt.xlabel('Value')
    plt.ylabel('Density')

    plt.subplot(2, 2, 3)
    sns.histplot(diastolic_bp_sample, kde=True, color='red', stat='density', bins=20)
    plt.title('Diastolic Blood Pressure for Men\nMean=80, SD=20')
    plt.xlabel('Blood Pressure')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample_size = 1000
    mean_unimodal = 100
    std_unimodal = 20

    mean_bp = 80
    std_bp = 20
    unimodal_sample = generate_unimodal_sample(mean_unimodal, std_unimodal, sample_size)
    multimodal_sample = generate_multimodal_sample(100, 130, std_unimodal, sample_size)
    diastolic_bp_sample = generate_unimodal_sample(mean_bp, std_bp, sample_size)

    plot_distributions(unimodal_sample, multimodal_sample, diastolic_bp_sample)

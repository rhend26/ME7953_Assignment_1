import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Dataset
normal_data = pd.read_csv('no_fault.csv', usecols=[0, 1])  # Normal condition vibration data
normal_data_filter = normal_data.dropna()
faulty_data = pd.read_csv('eccentricity.csv', usecols=[0, 1])  # Faulty condition vibration data
faulty_data_filter = faulty_data.dropna()


# Combine datasets for preprocessing
data = pd.concat([normal_data_filter, faulty_data_filter], axis=0)
labels = np.array([0] * len(normal_data_filter) + [1] * len(faulty_data_filter))  # 0: Normal, 1: Faulty

# Standardize the Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA
num_components = 3      # Change this number to experiment
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(data_scaled)

# Explained Variance of Each Component
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid',
         label='cumulative explained variance')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# Project Data onto First Two Principal Components
projected_data = principal_components[:, :2]

# Visualization of Projected Data
plt.figure(figsize=(8, 6))
for label, color in zip([0, 1], ['blue', 'red']):
    plt.scatter(projected_data[labels == label, 0], projected_data[labels == label, 1],
                label='Normal' if label == 0 else 'Faulty', alpha=0.7, edgecolor='k')
plt.title('Data Projection onto First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Reconstruction Error for Fault Detection
reconstructed_data = pca.inverse_transform(principal_components)
reconstruction_error = np.mean((data_scaled - reconstructed_data)**2, axis=1)

# Plot Reconstruction Error
plt.figure(figsize=(8, 6))
plt.hist(reconstruction_error[labels == 0], bins=30, alpha=0.7, label='Normal', color='blue', edgecolor='k')
plt.hist(reconstruction_error[labels == 1], bins=30, alpha=0.7, label='Faulty', color='red', edgecolor='k')
plt.axvline(np.percentile(reconstruction_error[labels == 0], 95), color='green', linestyle='--', label='Threshold')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Set Threshold and Detect Faults
threshold = np.percentile(reconstruction_error[labels == 0], 95)
predicted_labels = (reconstruction_error > threshold).astype(int)

# Evaluate Detection Performance
from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:\n", confusion_matrix(labels, predicted_labels))
print("\nClassification Report:\n", classification_report(labels, predicted_labels))

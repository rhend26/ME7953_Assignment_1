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

# Load additional fault datasets if available (example names)
surface_faults = pd.read_csv('surface_faults.csv', usecols=[0, 1]).dropna()
root_cracks = pd.read_csv('root_cracks.csv', usecols=[0, 1]).dropna()

# Combine datasets for preprocessing
data = pd.concat([normal_data_filter, faulty_data_filter, surface_faults, root_cracks], axis=0)

# Create labels for each fault type
labels = np.array([0] * len(normal_data_filter) + 
                  [1] * len(faulty_data_filter) + 
                  [2] * len(surface_faults) + 
                  [3] * len(root_cracks))  # 0: Normal, 1: Faulty, 2: Surface Faults, 3: Root Cracks

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

# Visualization of Projected Data for Multiple Fault Types
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'purple']  # Assign colors for each fault type
fault_labels = ['Normal', 'Eccentricity Fault', 'Surface Fault', 'Root Crack']

for label, color in zip(range(4), colors):
    plt.scatter(projected_data[labels == label, 0], projected_data[labels == label, 1], 
                label=fault_labels[label], alpha=0.7, edgecolor='k')

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

# Set Thresholds for Multi-Class Fault Detection
thresholds = [np.percentile(reconstruction_error[labels == i], 95) for i in range(4)]
predicted_labels = np.array([np.argmax([reconstruction_error[i] > thresholds[j] for j in range(4)]) for i in range(len(reconstruction_error))])

# Evaluate Detection Performance
from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:\n", confusion_matrix(labels, predicted_labels))
print("\nClassification Report:\n", classification_report(labels, predicted_labels))
print("\n--Now Performing Fault Detection using SVD ---\n")

from numpy.linalg import svd

# Perform SVD on standardized data
U, S, Vt = svd(data_scaled, full_matrices=False)
reconstructed_data_svd = np.dot(U, np.dot(np.diag(S), Vt))

# Compute Reconstruction Error for SVD
reconstruction_error_svd = np.mean((data_scaled - reconstructed_data_svd) ** 2, axis=1)

# Plot SVD Reconstruction Error
plt.figure(figsize=(8, 6))
plt.hist(reconstruction_error_svd[labels == 0], bins=30, alpha=0.7, label='Normal (SVD)', color='blue', edgecolor='k')
plt.hist(reconstruction_error_svd[labels == 1], bins=30, alpha=0.7, label='Faulty (SVD)', color='red', edgecolor='k')
plt.axvline(np.percentile(reconstruction_error_svd[labels == 0], 95), color='green', linestyle='--', label='Threshold (SVD)')
plt.title('Reconstruction Error Distribution (SVD)')
plt.xlabel('Reconstruction Error (SVD)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Set Threshold and Detect Faults using SVD
threshold_svd = np.percentile(reconstruction_error_svd[labels == 0], 95)
predicted_labels_svd = (reconstruction_error_svd > threshold_svd).astype(int)

# Evaluate SVD-Based Fault Detection Performance
print("\nConfusion Matrix (SVD):\n", confusion_matrix(labels, predicted_labels_svd))
print("\nClassification Report (SVD):\n", classification_report(labels, predicted_labels_svd))

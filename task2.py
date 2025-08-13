import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def load_and_preprocess_data(filepath):
    """Load and preprocess the customer purchase data"""
    data = pd.read_csv(filepath)
    
    # Example features - replace with your actual columns
    features = data[['Annual_Spending', 'Purchase_Frequency', 'Avg_Transaction_Value']]
    
    # Handle missing values if any
    features = features.dropna()
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, data

def find_optimal_clusters(data):
    """Determine optimal number of clusters using elbow method"""
    inertia = []
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    
    plt.tight_layout()
    plt.show()
    
    return k_range[np.argmax(silhouette_scores)]

def perform_clustering(data, k):
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

def analyze_clusters(original_data, clusters):
    """Analyze and visualize the clusters"""
    original_data['Cluster'] = clusters
    
    # Cluster statistics
    cluster_stats = original_data.groupby('Cluster').agg({
        'Annual_Spending': ['mean', 'median', 'std'],
        'Purchase_Frequency': ['mean', 'median'],
        'Avg_Transaction_Value': ['mean', 'median']
    })
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    for cluster in set(clusters):
        cluster_data = original_data[original_data['Cluster'] == cluster]
        plt.scatter(cluster_data['Annual_Spending'], 
                   cluster_data['Purchase_Frequency'], 
                   label=f'Cluster {cluster}')
    
    plt.title('Customer Segmentation')
    plt.xlabel('Annual Spending ($)')
    plt.ylabel('Purchase Frequency')
    plt.legend()
    plt.show()
    
    return original_data

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    filepath = 'customer_purchases.csv'  # Replace with your file path
    scaled_data, original_data = load_and_preprocess_data(filepath)
    
    # Step 2: Find optimal number of clusters
    optimal_k = find_optimal_clusters(scaled_data)
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Step 3: Perform clustering
    clusters, kmeans = perform_clustering(scaled_data, optimal_k)
    
    # Step 4: Analyze results
    clustered_data = analyze_clusters(original_data.copy(), clusters)
    
    # Save results
    clustered_data.to_csv('customer_segments.csv', index=False)
    print("\nClustering completed. Results saved to 'customer_segments.csv'")

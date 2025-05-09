import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

print("Dataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)
df = df[(df['Model Year'] >= 1990) & (df['Model Year'] <= 2025)]
print("\nShape after cleaning:", df.shape)

reduced_df = df[[
    'Make',
    'Model',
    'Model Year',
    'Electric Range',
    'Base MSRP',
    'County',
    'City',
    'Electric Vehicle Type'
]]

print("\nShape after reduction:", reduced_df.shape)
print(reduced_df.head())

scaler = MinMaxScaler()
cols_to_scale = ['Electric Range', 'Base MSRP']
reduced_df.loc[:, cols_to_scale] = scaler.fit_transform(reduced_df[cols_to_scale])

print("\nAfter normalization:")
print(reduced_df[cols_to_scale].describe())

range_bins = [0.0, 0.1, 0.5, 1.0]
range_labels = ['Low', 'Medium', 'High']
reduced_df.loc[:, 'Range Category'] = pd.cut(
    reduced_df['Electric Range'],
    bins=range_bins,
    labels=range_labels,
    include_lowest=True
)

print("\nElectric Range categories:")
print(reduced_df['Range Category'].value_counts())

reduced_df = reduced_df.dropna(subset=['Electric Range', 'Base MSRP', 'Model Year'])

X = reduced_df[['Electric Range', 'Base MSRP', 'Model Year']].copy()
X['Model Year'] = MinMaxScaler().fit_transform(X[['Model Year']])

inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
reduced_df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(
    reduced_df['Electric Range'],
    reduced_df['Base MSRP'],
    c=reduced_df['Cluster'],
    cmap='viridis',
    s=50
)
plt.xlabel('Electric Range (normalized)')
plt.ylabel('Base MSRP (normalized)')
plt.title('K-Means Clustering of Electric Vehicles')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

print("\nPreview with Clusters:")
print(reduced_df[['Make', 'Model', 'Model Year', 'Electric Range', 'Base MSRP', 'Cluster']].head(10))

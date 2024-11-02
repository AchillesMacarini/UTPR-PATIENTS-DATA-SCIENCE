import import_data
import matplotlib.pyplot as plt

rand_features = import_data.features.sample(n=10, random_state=1, axis=1)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 10))
axes = axes.flatten()

for i, coluna in enumerate(rand_features.columns):
    for classe in import_data.classes.unique():
        subset = import_data.data[import_data.classes == classe]
        class_name = import_data.general_health_data['GeneralHealth'][classe]
        axes[i].hist(subset[coluna], bins=30, alpha=0.5, label=f'{class_name}', density=True)
    
    axes[i].set_xlim(import_data.features[coluna].min(), import_data.features[coluna].max())
    axes[i].set_ylim(0, axes[i].get_ylim()[1])
    
    axes[i].set_title(f'{coluna}')
    axes[i].set_ylabel('Density')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()

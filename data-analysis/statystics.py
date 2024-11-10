import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import import_data
from scipy.stats import kurtosis, entropy

data = import_data.data
classes = import_data.classes
features = import_data.features

atributos_extraidos = {
    'Classe': [],
    'Média': [],
    'Desvio Padrão': [],
    # 'Coeficiente de Variação': [],
    # 'Curtose': [],
    # 'Entropia': [],
    # 'RMS': [],
    'Máximo': [],
    'Mínimo': [],
    'Mediana': []
}

# Calculando os atributos para cada coluna
for coluna in enumerate(features.columns, start=1):
    for classe in classes:
        subset = import_data.data[import_data.classes == classe]  # Subconjunto para a classe atual
        
        # Calculando os atributos
        media = subset.mean()
        desvio_padrao = subset.std()
        # coeficiente_variacao = desvio_padrao / media if media.empty else 0
        # curtose = kurtosis(subset)
        # entropia_valor = entropy(np.histogram(subset, bins=30)[0])
        # rms = np.sqrt(np.mean(np.square(subset)))
        maximo = subset.max()
        minimo = subset.min()
        mediana = subset.median()
        
        # Armazenando os resultados
        atributos_extraidos['Classe'].append(classe)
        atributos_extraidos['Média'].append(media)
        atributos_extraidos['Desvio Padrão'].append(desvio_padrao)
        # atributos_extraidos['Coeficiente de Variação'].append(coeficiente_variacao)
        # atributos_extraidos['Curtose'].append(curtose)
        # atributos_extraidos['Entropia'].append(entropia_valor)
        # atributos_extraidos['RMS'].append(rms)
        atributos_extraidos['Máximo'].append(maximo)
        atributos_extraidos['Mínimo'].append(minimo)
        atributos_extraidos['Mediana'].append(mediana)

# Convertendo o dicionário em um DataFrame
atributos_df = pd.DataFrame(atributos_extraidos)

# Criando uma figura com subplots para visualização
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))  # 3 linhas, 3 colunas
axes = axes.flatten()  # Flatten para iterar facilmente

# Listando os atributos para os gráficos
atributos = ['Média',
             'Desvio Padrão',
            #  'Coeficiente de Variação',
            #  'Curtose',
            #  'Entropia',
            #  'RMS',
             'Máximo',
             'Mínimo',
             'Mediana']

# Criando gráficos para cada atributo
for i, atributo in enumerate(atributos):
    for classe in classes:
        subset = atributos_df[atributos_df['Classe'] == classe]  # Subconjunto para a classe atual
        axes[i].plot(subset['Classe'], subset[atributo], marker='o', label=f'Classe {classe}')  # Gráfico de linha
    
    axes[i].set_title(f'{atributo} por Classe')
    axes[i].set_xlabel('Classe')
    axes[i].set_ylabel(atributo)
    axes[i].legend()
    axes[i].grid(True)

# Ajustando o layout
plt.tight_layout()
plt.show()  # Exibe os gráficos



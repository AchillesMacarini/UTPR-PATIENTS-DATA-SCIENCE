import import_data
import pandas as pd  # Para manipulação de dados
import numpy as np  # Para operações numéricas
import matplotlib.pyplot as plt  # Para visualização de dados
from sklearn.model_selection import train_test_split  # Para dividir os dados
from sklearn.neighbors import KNeighborsClassifier  # Para o classificador KNN
from matplotlib.colors import ListedColormap  # Para mapear cores

dados = import_data.data
classes = import_data.classes
features = import_data.features

dados['GeneralHealth'] = dados['GeneralHealth'].astype('category').cat.codes

correlacoes = pd.DataFrame(columns=['Atributo', 'Coeficiente de Correlação'])

# Calculando o coeficiente de correlação para cada coluna em relação à coluna ID
for coluna in dados.columns:
    if coluna != 'GeneralHealth':
        # Calculando a correlação com a coluna ID
        correlacao = dados[coluna].corr(dados['GeneralHealth'])
        correlacoes = correlacoes._append({'Atributo': coluna, 'Coeficiente de Correlação': correlacao}, ignore_index=True)

# Ordenando os coeficientes de correlação do maior para o menor
correlacoes['Coeficiente de Correlação'] = pd.to_numeric(correlacoes['Coeficiente de Correlação'], errors='coerce')
correlacoes = correlacoes.sort_values(by='Coeficiente de Correlação', ascending=False)

# Filtrando apenas valores finitos
correlacoes = correlacoes[np.isfinite(correlacoes['Coeficiente de Correlação'])]

# Obtendo os dois atributos com os maiores coeficientes de correlação
atributo_maior_correlacao = correlacoes.iloc[0]['Atributo']
atributo_segundo_maior_correlacao = correlacoes.iloc[1]['Atributo']

# Selecionando as colunas para o modelo
X = dados[[atributo_maior_correlacao, atributo_segundo_maior_correlacao]].values
y = dados['GeneralHealth'].values

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Solicitando o valor de k ao usuário
k = 8

# Criando o classificador KNN
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Avaliando a acurácia de TESTE
acuracia = knn.score(X_test, y_test)
print(f'Acurácia do modelo KNN com k={k}: {acuracia:.2f}')

# Criando um gráfico para visualizar a região de aceitação
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Definindo o passo inicial e mínimo para a grade de previsão
step = 80.0
min_step = 1.0

# Loop para encontrar o menor passo que gere uma grade adequada para plt.contourf
while True:
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    if xx.shape[0] > 1 and xx.shape[1] > 1:
        break
    step = max(step / 2, min_step)  # Reduzindo o passo, mas não menos que min_step

# Prevendo a classe para cada ponto na grade
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotando a região de aceitação
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta')))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', marker='o', label='Treinamento')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', marker='x', label='Teste')
plt.title('Região de Aceitação do KNN')
plt.xlabel(atributo_maior_correlacao)
plt.ylabel(atributo_segundo_maior_correlacao)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

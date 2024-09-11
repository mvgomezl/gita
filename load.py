import pandas as pd
from sklearn.datasets import load_iris

# Carga el conjunto de datos
iris = load_iris()

# Convierte los datos a un DataFrame de pandas
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

df.to_csv('df.csv', encoding = 'utf-8-sig', index = False)

# Visualiza las primeras filas del DataFrame
print('Aca se imprime el dataframe iris:')
print('')
print(df.head())

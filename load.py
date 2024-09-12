import pandas as pd
from sklearn.datasets import load_iris

# Carga el conjunto de datos
iris = load_iris()

# Convierte los datos a un DataFrame de pandas
df_new = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_new['target'] = iris.target

df_old = pd.read_csv('df.csv', sep = ',', decimal = '.', header = 0, encoding = 'utf-8')

df = pd.concat([df_old, df_new], ignore_index=True)

df.to_csv('df.csv', encoding = 'utf-8-sig', index = False)

# Visualiza las primeras filas del DataFrame
print('Aca se imprime el dataframe iris:')
print('')
print(df.head())

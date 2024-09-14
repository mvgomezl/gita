import pandas as pd
from sklearn.datasets import load_iris
import datetime

ahora = datetime.datetime.now()
print(f'La fecha es: {ahora}')

# Carga el conjunto de datos
iris = load_iris()

# Convierte los datos a un DataFrame de pandas
df_new = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_new['target'] = iris.target

print(f'shape new: {df_new.shape}')

df_old = pd.read_csv('df.csv', sep = ',', decimal = '.', header = 0, encoding = 'utf-8')

print(f'shape old: {df_old.shape}')

df = pd.concat([df_old, df_new], ignore_index=True)

df.to_csv('df.csv', encoding = 'utf-8-sig', index = False)

print(f'shape actual: {df.shape}')

# Visualiza las primeras filas del DataFrame
# print('Aca se imprime el dataframe iris:')
# print('')
# print(df.head())

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:46:50 2024

@author: marce
"""

# Como experto en programación Phyton 3 crear un código que permita validar y hacer proyecciones del modelo
# La variables consideradas para validar el modelo son 'GDP_USA', 'precio','precio_30', 'precio_2'

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

nuevo_directorio = "C:/Users/marce/Proyecto vino"

# Cambiar al nuevo directorio
os.chdir(nuevo_directorio)

# Verificar que el cambio se haya realizado
nuevo_directorio_actual = os.getcwd()
print("Nuevo directorio actual:", nuevo_directorio_actual)

ruta = 'C:/Users/marce/Proyecto vino/Ultima_data_OLS_42v_20230319.xlsx'

data = pd.read_excel(ruta)

######## HACER SUBSET ################################
# Se realizarón 3 subset

data_correlacion = data[['CANT_4c', 'precio',  'precio_2', 'precio_3', 'precio_4', 'precio_6', 'precio_7',
        'precio_8', 'precio_12', 'precio_14', 'GDP_USA', 'precio_30', 'precio_31']]

subset_data = data[['ID', 'CANT_6', 'CANT_10', 'precio_16','precio','precio_2', 'precio_12', 'GDP_USA', 'precio_30', 'CANT_IQ', 'CANT_4c']]


# Quiero transformar la variable objeto "FECHA" a datetime
# Convertir la columna "FECHA" a datetime
data['FECHA'] = pd.to_datetime(data['FECHA'], format='%Y-%m')

# quiero hacer un grafico de la regresion OLS

df = pd.DataFrame(subset_data)
# print(df)

# Quiero hacer un grafico de modelo de regresion lineal OLS
# gráfico de análisis multivariable

# # Aplicar logaritmo natural a las variables
df_log = np.log(df)

# # Dividir los datos en variables dependientes e independientes
y = df_log['CANT_4c']
X = df_log[['GDP_USA','precio_30', 'precio_2','precio_12']]
# # Agregar una constante al conjunto de variables independientes (intercepto)
X = sm.add_constant(X)

# # Crear el modelo de regresión lineal
modelo_OLS = sm.OLS(y, X).fit()

# # Imprimir un resumen del modelo
print('Resumen del Modelo OLS para la variable (y = 4 código arancelarios): \n',modelo_OLS.summary())

# Visualización del modelo
# Gráfico de valores observados vs predichos
y_pred = modelo_OLS.predict(X)
plt.scatter(y, y_pred)
plt.xlabel('Valores Observados (log)')
plt.ylabel('Valores Predichos (log)')
plt.title('Valores Observados vs Valores Predichos')
plt.show()

# Gráfico de residuos
residuos = modelo_OLS.resid
plt.scatter(y_pred, residuos)
plt.xlabel('Valores Predichos (log)')
plt.ylabel('Residuos (log)')
plt.title('Residuos vs Valores Predichos')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

print("")
print("ESTADISTICOS:")
mean_residuals = np.mean(modelo_OLS.resid)
print("Media de los residuos:", mean_residuals)

std_error_regression = np.sqrt(modelo_OLS.mse_resid)
print("Error estándar de la regresión:", std_error_regression)

variance_error = np.var(modelo_OLS.resid)
print("Varianza del error:", variance_error)

r_squared_adj = modelo_OLS.rsquared_adj
print("R-cuadrado ajustado:", r_squared_adj)

f_statistic = modelo_OLS.fvalue
print("Estadístico F:", f_statistic)
print("")

######################################################################################
############################# MATRIZ DE CORRELACION ##################################
######################################################################################

print("MATRIZ DE CORRELACION: ")

# Calcular la matriz de correlación
correlation_matrix = data_correlacion.corr()

# Crear una máscara triangular superior
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Configurar el tamaño de la figura
plt.figure(figsize=(19, 15))

# Crear el mapa de correlación utilizando Seaborn y aplicar la máscara triangular superior
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)

# Añadir título
plt.title("Mapa de correlación")

# Mostrar el mapa de correlación
plt.show()

###########################################################################################
################ GRAFICA PAIRWISE #######################################################
########################################################################################

# Selecciona solo las columnas que quieres incluir en el gráfico pairwise
variables = ['CANT_4c','GDP_USA','precio','precio_30', 'precio_2','precio_12']
pairwise_df = df_log[variables]


# Crea el gráfico pairwise
sns.pairplot(pairwise_df)
plt.show()

##############################################################################################
################## GRAFICO DE ANALISIS DE REGRESION LINEAL OLS ###############################
##############################################################################################

# Obtener las predicciones del modelo
predictions = modelo_OLS.predict(X)

# Graficar las predicciones contra los valores reales de la variable dependiente
plt.figure(figsize=(10, 6))
plt.scatter(predictions, y, alpha=0.5)  # Graficar puntos reales vs predichos
plt.plot(y, y, color='red', label='Línea de 45 grados')  # Graficar línea de 45 grados para comparación
plt.title('Modelo de Regresión Lineal OLS')
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.legend()
#plt.show()

###############################################################################################
###################### PREDICCIÓN #############################################################
###############################################################################################

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dividir los datos en conjuntos de entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal y ajustarlo a los datos de entrenamiento
modelo_OLS = sm.OLS(y_train, X_train).fit()

# Obtener las predicciones del modelo en los datos de prueba
predictions_test = modelo_OLS.predict(X_test)

# Calcular el error cuadrático medio en los datos de prueba
mse = mean_squared_error(y_test, predictions_test)
print('Error Cuadrático Medio en los datos de prueba:', mse)

# Graficar las predicciones contra los valores reales en los datos de prueba
plt.figure(figsize=(10, 6))
plt.scatter(predictions_test, y_test, alpha=0.5)  # Graficar puntos reales vs predichos
plt.plot(y_test, y_test, color='red', label='Línea de 45 grados')  # Graficar línea de 45 grados para comparación
plt.title('Modelo de Regresión Lineal OLS - Datos de Prueba')
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.legend()
plt.show()

# Hacer una proyección con el modelo entrenado
# Supongamos que tenemos nuevos datos X_new para predecir y_new
# Ajustamos estos nuevos datos (X_new) de la misma manera que ajustamos los datos de entrenamiento
# y luego utilizamos el modelo ajustado para predecir y_new
# predicciones_proyeccion = modelo_OLS.predict(X_new)



##############################################################
############## STEPWISE BACKWISE #############################
##############################################################
print("")
print("MODELO STAPWINSE")


def forward_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() 
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


#############################################################
################# BACKWISE ##################################
#############################################################

def backward_elimination(X, y, threshold=0.05):
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        p_values = model.pvalues.iloc[1:]  # Exclude the constant term
        max_p_value = p_values.max()
        if max_p_value > threshold:
            changed = True
            excluded_feature = p_values.idxmax()
            included.remove(excluded_feature)
            print('Dropped feature:', excluded_feature, 'with p-value:', max_p_value)
        if not changed:
            break
    return included

# Supongamos que 'subset_data' es tu conjunto de datos


# Modeleci´pn de backwise
# Realizar backward selection
selected_features = backward_elimination(X, y)

# Ajustar el modelo con las variables seleccionadas
X_selected = X[selected_features]
X_selected = sm.add_constant(X_selected)
modelo_backwise = sm.OLS(y, X_selected).fit()

# Imprimir un resumen del modelo backwise
print('Resumen del Modelo Backwise: \n', modelo_backwise.summary())



###############################################################################


# Supongamos que 'subset_data' es tu conjunto de datos


# Realizar análisis stepwise
selected_features = forward_selection(X, y)

# Ajustar el modelo con las variables seleccionadas
X_selected = X[selected_features]
X_selected = sm.add_constant(X_selected)
modelo_stepwise = sm.OLS(y, X_selected).fit()

# Imprimir un resumen del modelo stepwise
print('Resumen del Modelo Stepwise: \n', modelo_stepwise.summary())



########################






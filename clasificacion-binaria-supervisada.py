# importamos las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# dataset de ejemplo
data = {
    'horasEstudio': [2, 10, 5, 8, 1, 7, 4, 9],
    'nivelConocimiento': [3, 8, 6, 9, 2, 7, 4, 10],
    'asistencia': [60, 95, 80, 100, 40, 90, 70, 98],
    'promedioTareas': [50, 90, 70, 95, 30, 85, 65, 100],
    'tipoEstudiante': [1, 0, 1, 0, 1, 0, 1, 0],
    'resultado': [0, 1, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# preparación de los datos
X = df.drop('resultado', axis=1)
y = df['resultado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# entreno del modelo - ejm:arbol de decision
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# predicción y evaluación
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# predicción para un estudiante nuevo
nuevo_estudiante = [[6, 7, 85, 75, 1]]  # ejemplo de entrada
resultado = model.predict(nuevo_estudiante)
print("Aprobado" if resultado[0] == 1 else "No Aprobado")
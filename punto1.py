#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punto 1 (Python) - Análisis sobre `winequality-red.csv`.

Versión centrada en clasificación: EDA (resumen, matriz de correlación, pairplot)
y Random Forest para clasificación multiclasal con métricas (accuracy, precision,
recall, f1, ROC AUC), matriz de confusión e importancias de variables.

Para ejecutar (Windows cmd):
    python punto1.py
o
    py punto1.py

Requiere: pandas, numpy, matplotlib, seaborn, scikit-learn
Opcional: imbalanced-learn (para `RandomOverSampler`)
Instalar: pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
try:
    from imblearn.over_sampling import RandomOverSampler
except Exception:
    RandomOverSampler = None


def read_wine(path='winequality-red.csv'):
    # Intentar separador ';' (formato común en algunos CSVs) y fallback a ','
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path} en el directorio actual")
    try:
        df = pd.read_csv(path, sep=';')
        if df.shape[1] == 1:
            # probablemente separador incorrecto
            df = pd.read_csv(path, sep=',')
    except Exception:
        df = pd.read_csv(path, sep=',')
    return df


def save_corr_and_pairs(df):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de correlación')
    plt.tight_layout()
    plt.savefig('correlation_matrix_py.png')
    plt.close()

    # Pairplot (puede ser pesado; guardamos una versión)
    try:
        sns.pairplot(df, diag_kind='kde', plot_kws={'s':10})
        plt.savefig('pairs_plot_py.png')
        plt.close()
    except Exception:
        # si falla por tamaño, muestreamos un subset
        sns.pairplot(df.sample(n=min(300, len(df))), diag_kind='kde', plot_kws={'s':20})
        plt.savefig('pairs_plot_py_sample.png')
        plt.close()


def fit_and_evaluate(df):
    # Análisis de regresión removido. Esta función existe solo como placeholder
    # para mantener compatibilidad si se desea reactivar análisis de regresión.
    print('Análisis de regresión eliminado en esta versión. Use fit_random_forest para clasificación.')


def fit_random_forest(df):
    """Ajusta Random Forest para clasificación multiclasal sobre `quality`.
    Guarda métricas (accuracy, precision, recall, f1, auc), matriz de confusión y
    feature importances en archivos en el directorio actual.
    """
    if 'quality' not in df.columns:
        raise ValueError('No se encontró la columna `quality` en el dataset')

    X = df.drop(columns=['quality'])
    y = df['quality']

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Escalado (opcional para RF no es obligatorio pero mantenemos consistencia para otros modelos)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Balanceo SOLO en training si imblearn está disponible
    if RandomOverSampler is not None:
        ros = RandomOverSampler(random_state=42)
        X_train_bal, y_train_bal = ros.fit_resample(X_train_s, y_train)
    else:
        X_train_bal, y_train_bal = X_train_s, y_train

    # Ajustar Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)

    # Predicciones
    y_pred = rf.predict(X_test_s)
    y_proba = rf.predict_proba(X_test_s)

    # Binarizar para AUC multiclase
    classes = np.sort(df['quality'].unique())
    try:
        y_test_bin = label_binarize(y_test, classes=classes)
        auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    except Exception:
        auc = np.nan

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'ROC_AUC_macro': auc
    }

    # Guardar métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('rf_classification_metrics_py.csv', index=False)

    # Guardar reporte de clasificación completo
    report = classification_report(y_test, y_pred, zero_division=0)
    with open('rf_classification_report.txt', 'w') as f:
        f.write(report)

    # Matriz de confusión y plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de confusión - Random Forest')
    plt.tight_layout()
    plt.savefig('rf_confusion_matrix_py.png')
    plt.close()

    # Importancias de características
    fi = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
    fi.to_csv('rf_feature_importances_py.csv', index=False)

    # Mostrar por consola resumen
    print('\nRandom Forest - métricas:')
    print(metrics_df.T)
    print('\nClassification report guardado en rf_classification_report.txt')
    print('Feature importances guardadas en rf_feature_importances_py.csv')


def main():
    print('Cargando datos...')
    df = read_wine('winequality-red.csv')
    print('Dimensiones:', df.shape)
    print('\nResumen:')
    print(df.describe())

    print('\nGuardando matriz de correlación y pairplot...')
    save_corr_and_pairs(df)

    print('\nAjustando modelos y evaluando...')
    fit_and_evaluate(df)
    print('\nAjustando Random Forest (clasificación) y calculando métricas...')
    fit_random_forest(df)


if __name__ == '__main__':
    main()

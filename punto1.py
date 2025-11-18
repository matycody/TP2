#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punto 1 (Python) - Análisis sobre `winequality-red.csv`.

Genera: EDA (resumen, matriz de correlación, pairplot), ajustes de modelos
OLS, Ridge (CV) y Lasso (CV), métricas (MSE, R2), coeficientes y gráficos
de residuos. Basado en la implementación en R en `punto1.R`.

Para ejecutar (Windows cmd):
    python punto1.py
o
    py punto1.py

Requiere: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels
Instalar: pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


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
    if 'quality' not in df.columns:
        raise ValueError('No se encontró la columna `quality` en el dataset')

    X = df.drop(columns=['quality'])
    y = df['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # OLS (statsmodels) para coeficientes en escala original
    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_sm).fit()

    # Predicción OLS
    X_test_sm = sm.add_constant(X_test)
    y_pred_ols = ols_model.predict(X_test_sm)

    # Regresión lineal (sklearn) como referencia
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Escalado para modelos penalizados
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # RidgeCV con grid de alphas
    alphas = np.logspace(-6, 6, 50)
    ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)
    ridge_cv.fit(X_train_s, y_train)
    y_pred_ridge = ridge_cv.predict(X_test_s)

    # LassoCV
    lasso_cv = LassoCV(alphas=None, cv=5, max_iter=5000, random_state=1)
    lasso_cv.fit(X_train_s, y_train)
    y_pred_lasso = lasso_cv.predict(X_test_s)

    # Métricas
    results = []
    for name, y_pred in [('OLS', y_pred_ols), ('LinearRegression', y_pred_lr),
                         ('Ridge', y_pred_ridge), ('Lasso', y_pred_lasso)]:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({'Modelo': name, 'MSE': mse, 'R2': r2})

    results_df = pd.DataFrame(results)
    results_df.to_csv('model_comparison_py.csv', index=False)

    # Guardar coeficientes: OLS (original scale)
    ols_coefs = pd.DataFrame({'term': ols_model.params.index, 'coef': ols_model.params.values})
    ols_coefs.to_csv('ols_coefficients_py.csv', index=False)

    # Convertir coef de Ridge y Lasso a escala original
    def to_original_scale(coef_scaled, intercept_scaled, scaler, X_train_mean):
        # coef_scaled corresponde a features estandarizadas
        scale = scaler.scale_
        mean = X_train_mean
        coef_orig = coef_scaled / scale
        intercept_orig = intercept_scaled - np.sum((coef_scaled * mean) / scale)
        return coef_orig, intercept_orig

    # Ridge
    ridge_coef_s = ridge_cv.coef_
    ridge_inter_s = ridge_cv.intercept_
    ridge_coef_orig, ridge_inter_orig = to_original_scale(ridge_coef_s, ridge_inter_s, scaler, X_train.mean(axis=0))
    ridge_coefs_df = pd.DataFrame({'term': X.columns, 'coef_scaled': ridge_coef_s, 'coef_orig': ridge_coef_orig})
    ridge_coefs_df['intercept_orig'] = ridge_inter_orig
    ridge_coefs_df.to_csv('ridge_coefficients_py.csv', index=False)

    # Lasso
    lasso_coef_s = lasso_cv.coef_
    lasso_inter_s = lasso_cv.intercept_
    lasso_coef_orig, lasso_inter_orig = to_original_scale(lasso_coef_s, lasso_inter_s, scaler, X_train.mean(axis=0))
    lasso_coefs_df = pd.DataFrame({'term': X.columns, 'coef_scaled': lasso_coef_s, 'coef_orig': lasso_coef_orig})
    lasso_coefs_df['intercept_orig'] = lasso_inter_orig
    lasso_coefs_df.to_csv('lasso_coefficients_py.csv', index=False)

    # Gráficos de residuos
    residuals = {
        'OLS': y_test - y_pred_ols,
        'LinearRegression': y_test - y_pred_lr,
        'Ridge': y_test - y_pred_ridge,
        'Lasso': y_test - y_pred_lasso,
    }

    plt.figure(figsize=(14,4))
    for i, (name, res) in enumerate(residuals.items(), 1):
        plt.subplot(1,4,i)
        sns.histplot(res, kde=True, bins=20)
        plt.title(f'Resid. {name}')
    plt.tight_layout()
    plt.savefig('residuals_histograms_py.png')
    plt.close()

    # Residuals vs Fitted (usar OLS, Ridge, Lasso)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.scatter(y_pred_ols, residuals['OLS'], s=20)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title('OLS: Ajustado vs Residuo')
    plt.subplot(1,3,2)
    plt.scatter(y_pred_ridge, residuals['Ridge'], s=20)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title('Ridge: Ajustado vs Residuo')
    plt.subplot(1,3,3)
    plt.scatter(y_pred_lasso, residuals['Lasso'], s=20)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title('Lasso: Ajustado vs Residuo')
    plt.tight_layout()
    plt.savefig('residuals_vs_fitted_py.png')
    plt.close()

    # Información adicional: alphas
    extra = {
        'ridge_alpha': ridge_cv.alpha_,
        'lasso_alpha': lasso_cv.alpha_
    }
    extra_df = pd.DataFrame([extra])
    extra_df.to_csv('model_alphas_py.csv', index=False)

    print('\nResultados guardados en archivos CSV/PNG en el directorio actual.')
    print(results_df)
    print('\nRidge alpha elegido:', ridge_cv.alpha_)
    print('Lasso alpha elegido:', lasso_cv.alpha_)


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


if __name__ == '__main__':
    main()

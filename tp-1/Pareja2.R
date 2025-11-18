# TP1 - Inferencia Estadística y Reconocimiento de Patrones

# Parte de regresión en R

# Cargamos las librerias
library(tidyverse)
library(corrplot)
library(glmnet)
library(GGally)

# Empezamos borrando todas las variables
rm(list = ls())

# Cargamos el dataset
vinos <- read.csv2("winequality-red.csv",dec=".")

# Miramos los nombres de las variables y vemos una vista rapida de la estructura del dataset
names(vinos)
head(vinos)
attach(vinos)

# Estadísticas descriptivas de cada variable
describe <- summary(vinos)
print(describe)

# Matriz de correlación y dispersión
corrplot(cor(vinos), method = 'color', tl.cex = 0.7)
cor_matrix <- cor(vinos)
ggpairs(vinos, columns = 1:12, title = "Matriz de correlaciones y dispersiC3n")
cor(vinos$quality, vinos[, 1:12])

# Separamos las variables explicativas de la variable respuesta
X <- vinos[, !(names(vinos) %in% c('quality'))]
y <- vinos$quality

# Armamos los conjuntos de entrenamiento y prueba en una relacion 80/20
set.seed(1)
train_idx <- sample(seq_len(nrow(vinos)), size = 0.8 * nrow(vinos))
train <- vinos[train_idx, ]
test <- vinos[-train_idx, ]

# Creo la matriz de diseño y vector de respuestas 
X_train <- as.matrix(train[, !(names(train) %in% c('quality'))])
y_train <- train$quality
X_test <- as.matrix(test[, !(names(test) %in% c('quality'))])
y_test <- test$quality

# Regresión lineal múltiple 
lm_fit <- lm(quality ~ ., data = train)
# Hacemos un summary del modelo
summary(lm_fit)

# El modelo lineal múltiple es estadísticamente significativo (p < 2.2e-16),
# lo que indica que al menos una variable predictora tiene
# asociaciC3n con la calidad del vino. El RB2 es de 0.3536, lo que implica que
# el modelo explica aproximadamente el 35% de la variabilidad en la respuesta.

# Las variables con mayor evidencia estadística (p < 0.001) son alcohol, 
# volatile y sulphates.

# Las variables con poca evidencia estadística (p > 0.05) son:
# fixed.acidity, citric.acid, residual.sugar, density, pH y free.sulfur.dioxide.

# Calculamos los predichos del conjunto de prueba
lm_pred <- predict(lm_fit, newdata = test)

# Metricas de evaluación del modelo 
cat('RegresiC3n lineal mC:ltiple:\n')
cat('Error CuadrC!tico Medio (MSE):', mean((test$quality - lm_pred)^2), '\n')
cat('R2:', summary(lm_fit)$r.squared, '\n')

# Graficos de residuos

residuos <- resid(lm_fit)
ajustados <- fitted(lm_fit)

# Histograma de Residuos
hist(residuos,
     breaks = 30,
     col = "steelblue",
     main = "DistribuciC3n de residuos",
     xlab = "Residuos")

# Gráfico de dispersión
plot(ajustados, residuos,
     xlab = "Valores ajustados",
     ylab = "Residuos",
     main = "Residuos vs Valores ajustados",
     pch = 20, col = "darkred")
abline(h = 0, lty = 2, col = "gray")

# Ridge 

cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)
best_lambda_ridge <- cv_ridge$lambda.min
ridge_pred <- predict(cv_ridge, s = best_lambda_ridge, newx = X_test)

# Metricas de evaluación del modelo 
cat('\nRidge:\n')
r2_ridge <- 1 - sum((y_test - ridge_pred)^2) / sum((y_test - mean(y_test))^2)
cat("Error CuadrC!tico Medio (MSE): ", mean((y_test - ridge_pred)^2))
cat('R2:', r2_ridge, '\n')

# Graficos de residuos
residuos_ridge <- y_test-ridge_pred

# Histograma
hist(residuos_ridge,
     breaks = 30,
     col = "tomato",
     main = "Residuos del modelo penalizado",
     xlab = "Residuos")

# Valores ajustados vs Residuos
plot(ridge_pred, residuos_ridge,
     xlab = "Valores ajustados",
     ylab = "Residuos",
     main = "Residuos vs Valores ajustados (penalizado)",
     pch = 20, col = "blue")
abline(h = 0, lty = 2, col = "gray")

# LASSO 
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)
best_lambda_lasso <- cv_lasso$lambda.min
lasso_pred <- predict(cv_lasso, s = best_lambda_lasso, newx = X_test)

# Metricas de evaluación del modelo 
r2_lasso <- 1 - sum((y_test - lasso_pred)^2) / sum((y_test - mean(y_test))^2)
cat('\nLASSO:\n')
cat("Error CuadrC!tico Medio (MSE): ", mean((y_test - lasso_pred)^2))
cat('R2:', r2_lasso, '\n')

# Graficos de residuos
residuos_lasso <- y_test-lasso_pred

# Histograma
hist(residuos_lasso,
     breaks = 30,
     col = "tomato",
     main = "Residuos del modelo penalizado",
     xlab = "Residuos")

# Valores ajustados vs Residuos
plot(lasso_pred, residuos_lasso,
     xlab = "Valores ajustados",
     ylab = "Residuos",
     main = "Residuos vs Valores ajustados (penalizado)",
     pch = 20, col = "blue")
abline(h = 0, lty = 2, col = "gray")

# Comparación de residuos: Boxplot de los tres modelos
residuos_df <- data.frame(
     Residuo = c(as.numeric(residuos), as.numeric(residuos_ridge), as.numeric(residuos_lasso)),
     Modelo = factor(
          rep(c("Lineal", "Ridge", "LASSO"),
                    times = c(length(residuos), length(residuos_ridge), length(residuos_lasso)))
     )
)

boxplot(Residuo ~ Modelo, data = residuos_df,
                    col = c("steelblue", "tomato", "orange"),
                    main = "Comparación de residuos por modelo",
                    ylab = "Residuos",
                    xlab = "Modelo")

# Calidad real vs valores predichos por modelo 
calidad_predicho_df <- data.frame(
     Real = rep(y_test, 3),
     Predicho = c(as.numeric(lm_pred), as.numeric(ridge_pred), as.numeric(lasso_pred)),
     Modelo = factor(
          rep(c("Lineal", "Ridge", "LASSO"),
                    each = length(y_test))
     )
)
boxplot(Predicho ~ Real + Modelo, data = calidad_predicho_df,
                    col = c("steelblue", "tomato", "orange"),
                    main = "Valores predichos agrupados por calidad real y modelo",
                    ylab = "Valor predicho",
                    xlab = "Calidad real (por modelo)",
                    las = 2)


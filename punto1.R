# Punto 1 - Análisis sobre `winequality-red.csv`
# Basado en el ejemplo de `tp-1/Pareja2.R` (regresión y modelos penalizados)

## Requisitos: tidyverse, glmnet, GGally, corrplot
pkgs <- c('tidyverse','glmnet','GGally','corrplot')
install_if_missing <- function(p) if(!requireNamespace(p, quietly=TRUE)) install.packages(p)
invisible(sapply(pkgs, install_if_missing))

library(tidyverse)
library(glmnet)
library(GGally)
library(corrplot)

set.seed(1)

# Cargar datos (archivo debe estar en la misma carpeta que este script)
data_path <- 'winequality-red.csv'
if(!file.exists(data_path)){
  stop(paste('No se encontró', data_path, 'en el directorio actual.'))
}

# En este dataset los separadores pueden variar; intentar leer con read.csv y read.csv2
tryCatch({
  vinos <- read.csv(data_path, dec='.')
}, error = function(e){
  vinos <<- read.csv2(data_path, dec='.')
})

cat('Filas, columnas:', dim(vinos), '\n')
cat('Nombres de variables:', paste(names(vinos), collapse=', '), '\n\n')

# Resumen y correlaciones
print(summary(vinos))
png('correlation_matrix.png', width=800, height=600)
corrplot(cor(vinos), method='color', tl.cex = 0.7)
dev.off()

# Pares de variables (guarda como imagen para revisar)
png('pairs_plot.png', width=1200, height=900)
GGally::ggpairs(vinos)
dev.off()

# Preparar datos: separar X e y
if(!'quality' %in% names(vinos)) stop('No se encontró la variable `quality` en el dataset')
X <- vinos %>% select(-quality)
y <- vinos$quality

# Split 80/20
set.seed(1)
train_idx <- sample(seq_len(nrow(vinos)), size = 0.8 * nrow(vinos))
train <- vinos[train_idx, ]
test <- vinos[-train_idx, ]

X_train <- as.matrix(train %>% select(-quality))
y_train <- train$quality
X_test <- as.matrix(test %>% select(-quality))
y_test <- test$quality

# 1) Regresión lineal múltiple
lm_fit <- lm(quality ~ ., data = train)
lm_sum <- summary(lm_fit)
cat('\n--- Regresión lineal ---\n')
print(lm_sum)

lm_pred <- predict(lm_fit, newdata = test)
lm_mse <- mean((y_test - lm_pred)^2)
lm_r2 <- 1 - sum((y_test - lm_pred)^2)/sum((y_test - mean(y_test))^2)
cat(sprintf('\nLM MSE: %.4f, R2: %.4f\n', lm_mse, lm_r2))

# Guardar coeficientes
write.csv(data.frame(term = names(coef(lm_fit)), estimate = as.numeric(coef(lm_fit))), 'lm_coefficients.csv', row.names = FALSE)

# 2) Ridge (alpha = 0)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)
best_lambda_ridge <- cv_ridge$lambda.min
ridge_pred <- predict(cv_ridge, s = best_lambda_ridge, newx = X_test)
ridge_mse <- mean((y_test - ridge_pred)^2)
ridge_r2 <- 1 - sum((y_test - ridge_pred)^2)/sum((y_test - mean(y_test))^2)
cat(sprintf('\nRidge lambda*: %.5f | MSE: %.4f, R2: %.4f\n', best_lambda_ridge, ridge_mse, ridge_r2))

# 3) LASSO (alpha = 1)
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)
best_lambda_lasso <- cv_lasso$lambda.min
lasso_pred <- predict(cv_lasso, s = best_lambda_lasso, newx = X_test)
lasso_mse <- mean((y_test - lasso_pred)^2)
lasso_r2 <- 1 - sum((y_test - lasso_pred)^2)/sum((y_test - mean(y_test))^2)
cat(sprintf('\nLASSO lambda*: %.5f | MSE: %.4f, R2: %.4f\n', best_lambda_lasso, lasso_mse, lasso_r2))

# Guardar coeficientes de glmnet
ridge_coefs <- as.matrix(coef(cv_ridge, s = best_lambda_ridge))
lasso_coefs <- as.matrix(coef(cv_lasso, s = best_lambda_lasso))
write.csv(data.frame(term = rownames(ridge_coefs), ridge = ridge_coefs[,1]), 'ridge_coefficients.csv', row.names = FALSE)
write.csv(data.frame(term = rownames(lasso_coefs), lasso = lasso_coefs[,1]), 'lasso_coefficients.csv', row.names = FALSE)

# 4) Gráficos de residuos y comparación
png('residuals_histograms.png', width=1200, height=400)
par(mfrow=c(1,3))
hist(y_test - lm_pred, main='Residuos LM', xlab='Residuo', col='steelblue', breaks=20)
hist(y_test - ridge_pred, main='Residuos Ridge', xlab='Residuo', col='tomato', breaks=20)
hist(y_test - lasso_pred, main='Residuos LASSO', xlab='Residuo', col='orange', breaks=20)
dev.off()

png('residuals_vs_fitted.png', width=1200, height=400)
par(mfrow=c(1,3))
plot(lm_pred, y_test - lm_pred, main='LM: Ajustado vs Residuo', xlab='Ajustado', ylab='Residuo', pch=20)
abline(h=0, lty=2)
plot(as.numeric(ridge_pred), y_test - ridge_pred, main='Ridge: Ajustado vs Residuo', xlab='Ajustado', ylab='Residuo', pch=20)
abline(h=0, lty=2)
plot(as.numeric(lasso_pred), y_test - lasso_pred, main='LASSO: Ajustado vs Residuo', xlab='Ajustado', ylab='Residuo', pch=20)
abline(h=0, lty=2)
dev.off()

# 5) Comparación en data frame
results_summary <- data.frame(
  Modelo = c('Linear', 'Ridge', 'LASSO'),
  Lambda = c(NA, best_lambda_ridge, best_lambda_lasso),
  MSE = c(lm_mse, ridge_mse, lasso_mse),
  R2 = c(lm_r2, ridge_r2, lasso_r2)
)
write.csv(results_summary, 'model_comparison.csv', row.names = FALSE)

cat('\nAnálisis completado. Archivos generados:\n')
cat('- correlation_matrix.png\n- pairs_plot.png\n- lm_coefficients.csv\n- ridge_coefficients.csv\n- lasso_coefficients.csv\n- residuals_histograms.png\n- residuals_vs_fitted.png\n- model_comparison.csv\n')

cat('\nPara ejecutar: en Windows (cmd) ejecutar `Rscript punto1.R` desde la carpeta del TP2.\n')

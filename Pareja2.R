# Punto 3 - TP: Agrupamiento de exoplanetas
# Requiere: HSAUR2, cluster, factoextra, ggplot2, plotly (opcional)
# Instalar paquetes si no están:
pkgs <- c("HSAUR2","cluster","factoextra","ggplot2","plotly","gridExtra")
install_if_missing <- function(p) if(!requireNamespace(p, quietly=TRUE)) install.packages(p)
invisible(sapply(pkgs, install_if_missing))

library(HSAUR2)    # contiene data("planets")
library(cluster)   # pam, silhouette
library(factoextra) # funciones útiles para clustering y graficar
library(ggplot2)
library(plotly)
library(gridExtra)

set.seed(123) # reproducibilidad

# Cargar datos
data("planets", package = "HSAUR2")
# Revisar estructura
str(planets)
head(planets)

# La consigna pide usar masa (mass), periodo (period) y excentricidad (eccen)
df <- planets[, c("mass","period","eccen")]
summary(df)

# a) Estandarización y gráfico tridimensional
# Escalar (media 0, sd 1)
df_scaled <- scale(df)

# Gráfico 3D (plotly interactivo)
p3d <- plot_ly(x = df_scaled[,1], y = df_scaled[,2], z = df_scaled[,3],
               type = "scatter3d", mode = "markers",
               marker = list(size = 3)) %>%
  layout(scene = list(xaxis = list(title = "mass (scaled)"),
                      yaxis = list(title = "period (scaled)"),
                      zaxis = list(title = "eccen (scaled)")))
# Mostrar en RStudio Viewer
print(p3d)

# También un scatter pairs para ver relaciones
pairs(df_scaled, pch = 20, main = "Pairs - variables escaladas")

# b) Aplicar kmeans con 4 centroides,  nstart grande para buscar mínimo
set.seed(123)
k4 <- kmeans(df_scaled, centers = 4, nstart = 50, iter.max = 100)
k4$tot.withinss   # valor de la función objetivo
k4$withinss
table(k4$cluster)

# Visualizar clusters (PCA + colores)
fviz_cluster(k4, data = df_scaled, palette = "jco",
             geom = "point", ellipse.type = "convex",
             ggtheme = theme_minimal()) + ggtitle("K-means k=4")

# c) Repetir varias veces y comparar
# Repetir kmeans 20 veces y registrar tot.withinss
set.seed(123)
runs <- 20
totw <- numeric(runs)
models <- vector("list", runs)
for(i in 1:runs){
  km <- kmeans(df_scaled, centers = 4, nstart = 25)
  totw[i] <- km$tot.withinss
  models[[i]] <- km
}
data.frame(run = 1:runs, tot.withinss = totw)
# Mostrar mínimo y cuál run
which.min(totw); min(totw)

# d) Ejecutar K-means eligiendo centroides a partir de algún criterio
# Ejemplo: inicializar con 4 observaciones seleccionadas (kmeans + centers iniciales)
# Elegimos 4 observaciones con quantiles en mass y period para dar variedad:
init_idx <- c(which.min(df$mass), which.max(df$mass),
              which.min(df$period), which.max(df$period))
init_centers <- df_scaled[unique(init_idx), ]
set.seed(123)
km_init <- kmeans(df_scaled, centers = init_centers, iter.max = 100)
km_init$tot.withinss
fviz_cluster(km_init, data = df_scaled, ggtitle("Kmeans - centroides iniciales seleccionados"))

# e) K-medoides (pam) usando package cluster
# pam requiere número de clusters k
set.seed(123)
pam4 <- pam(df_scaled, k = 4)
pam4$medoids
fviz_cluster(pam4, geom = "point", ellipse.type = "convex") + ggtitle("PAM k=4")

# f) Método del codo (wss) para identificar cantidad de clusters
wss <- function(k) {
  kmeans(df_scaled, centers = k, nstart = 25)$tot.withinss
}
k.values <- 1:10
wss_values <- sapply(k.values, wss)
plot(k.values, wss_values, type="b", pch=19, frame=FALSE,
     xlab="Número de clusters K", ylab="Tot. within-cluster SS",
     main="Método del codo (WSS)")

# Alternativa con factoextra: fviz_nbclust
fviz_nbclust(df_scaled, kmeans, method = "wss") + ggtitle("Elbow method (factoextra)")

# Método silhouette
fviz_nbclust(df_scaled, kmeans, method = "silhouette") + ggtitle("Silhouette method")

# Calcular silhouette para k = 2..10 y mostrar promedios
sil_width <- sapply(2:10, function(k){
  pam.res <- pam(df_scaled, k = k)
  mean(silhouette(pam.res$clustering, dist(df_scaled))[,3])
})
plot(2:10, sil_width, type="b", pch=19, xlab="k", ylab="Average silhouette width",
     main="Silhouette promedio (usando PAM)")

# g) Usar método Silhouette para identificar cantidad de clusters:
# Mostrar cuál k tiene mayor silhouette promedio
best_k_sil <- which.max(sil_width) + 1  # porque empezamos en 2
best_k_sil
cat("Mejor k según silhouette (entre 2 y 10):", best_k_sil, "\n")

# h) Conclusión - ejemplo de cómo obtener y reportar resultados finales:
# Usar kmeans y pam con el k sugerido y mostrar centros/medoides y tamaño de clusters
k_recom <- best_k_sil
set.seed(123)
final_km <- kmeans(df_scaled, centers = k_recom, nstart = 50)
final_pam <- pam(df_scaled, k = k_recom)

# Tablas resumen
cat("Kmeans - tamaños de clusters:\n"); print(table(final_km$cluster))
cat("PAM - tamaños de clusters:\n"); print(table(final_pam$clustering))

# Mostrar centroides (desescalados) para interpretación:
centroids_scaled <- final_km$centers
# volver a la escala original:
center_means <- attr(df_scaled, "scaled:center")
center_sds   <- attr(df_scaled, "scaled:scale")
centroids_orig <- sweep(centroids_scaled, 2, center_sds, FUN="*")
centroids_orig <- sweep(centroids_orig, 2, center_means, FUN="+")
print("Centroides (valores en escala original):")
print(centroids_orig)

# Guardar plots si es necesario:
ggsave("elbow_wss.png")
ggsave("silhouette_plot.png")

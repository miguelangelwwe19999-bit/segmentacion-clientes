"""
Dashboard - Caso 5: Segmentación de clientes de una plataforma de comercio electrónico
Reproduce el mismo proceso del notebook de Google Colab:
limpieza -> selección de variables -> escalado -> K-Means / Jerárquico / DBSCAN -> perfiles.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentación de clientes", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------
# Parámetros del modelo (los mismos del notebook)
# ---------------------------------------------------------------------
K = 3
RANDOM_STATE = 42
EPS, MIN_SAMPLES = 2.0, 10
CSV_LOCAL = Path(__file__).parent / "clientes_ecommerce.csv"

FEATURES = [
    "Edad", "Antigüedad meses", "Compras últimos 6 meses",
    "Gasto total últimos 6 meses", "Valor promedio", "Días desde última compra",
    "Número de productos promedio", "Descuento promedio en porcentaje",
    "Uso cupones en porcentaje", "Visitas mensuales", "Tiempo promedio en minutos",
]
CATEGORICAS = ["Categoría más frecuente", "Canal preferido"]
VARS_PERFIL = [
    "Compras últimos 6 meses", "Frecuencia mensual", "Gasto total últimos 6 meses",
    "Valor promedio", "Días desde última compra", "Visitas mensuales",
    "Tiempo promedio en minutos", "Descuento promedio en porcentaje", "Uso cupones en porcentaje",
]
VARS_PRINCIPALES = ["Gasto total últimos 6 meses", "Valor promedio", "Compras últimos 6 meses", "Visitas mensuales"]
COLORES = {
    "Alto valor por compra": "#7b3294",
    "Frecuentes sensibles a promociones": "#1b9e77",
    "Ocasionales de baja interacción": "#e6ab02",
}


# ---------------------------------------------------------------------
# Proceso completo (se calcula una vez y queda en caché)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Procesando datos y entrenando modelos...")
def procesar(df: pd.DataFrame):
    res = {"df_original": df}

    # Punto 5: excluir el identificador
    df_matriz = df.drop(columns=["Id cliente"])

    # Punto 6: duplicados, inconsistencias y nulos
    limpieza = {"Filas duplicadas": int(df_matriz.duplicated().sum())}
    dfc = df_matriz.drop_duplicates().copy()
    m_edad = (dfc["Edad"] < 18) | (dfc["Edad"] > 100)
    m_cup = dfc["Uso cupones en porcentaje"] > 100
    limpieza["Edades inconsistentes"] = int(m_edad.sum())
    limpieza["Cupones > 100 %"] = int(m_cup.sum())
    dfc.loc[m_edad, "Edad"] = np.nan
    dfc.loc[m_cup, "Uso cupones en porcentaje"] = np.nan
    m_gasto = dfc["Gasto total últimos 6 meses"] == 0
    limpieza["Gasto total = 0"] = int(m_gasto.sum())
    dfc.loc[m_gasto, "Gasto total últimos 6 meses"] = (dfc["Compras últimos 6 meses"] * dfc["Valor promedio"])[m_gasto]
    limpieza["Nulos imputados"] = int(dfc.isnull().sum().sum())
    for col in dfc.select_dtypes(include=[np.number]).columns:
        dfc[col] = dfc[col].fillna(dfc[col].median())
    for col in dfc.select_dtypes(include=["object", "category", "string"]).columns:
        dfc[col] = dfc[col].fillna(dfc[col].mode()[0])
    res["limpieza"] = limpieza
    res["corr_frecuencia"] = float(dfc["Compras últimos 6 meses"].corr(dfc["Frecuencia mensual"]))

    # Puntos 7, 8 y 9: selección, one-hot y estandarización
    X_cat = pd.get_dummies(dfc[CATEGORICAS], drop_first=True)
    scaler = StandardScaler()
    X_num = pd.DataFrame(scaler.fit_transform(dfc[FEATURES]), columns=FEATURES, index=dfc.index)
    X = pd.concat([X_num, X_cat], axis=1)
    X_arr = np.asarray(X, dtype=float)

    # Punto 11: codo y silueta
    codo = []
    for k in range(2, 9):
        km_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        codo.append({"K": k, "Inercia": km_k.inertia_, "Silueta": silhouette_score(X, km_k.labels_)})
    res["codo"] = pd.DataFrame(codo)

    # Modelos
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    dfc["Cluster_KMeans"] = kmeans.fit_predict(X)
    dfc["Cluster_Jerarquico"] = AgglomerativeClustering(n_clusters=K).fit_predict(X)
    dfc["Cluster_DBSCAN"] = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X)

    m_db = dfc["Cluster_DBSCAN"].values != -1
    res["comparativa"] = pd.DataFrame({
        "Técnica": ["K-Means", "Jerárquico (Ward)", "DBSCAN"],
        "Parámetros": [f"n_clusters={K}, random_state={RANDOM_STATE}, n_init=10",
                       f"n_clusters={K}, enlace Ward", f"eps={EPS}, min_samples={MIN_SAMPLES}"],
        "Grupos": [K, K, len(set(dfc.loc[m_db, "Cluster_DBSCAN"]))],
        "Clientes por grupo": [
            " / ".join(map(str, dfc["Cluster_KMeans"].value_counts().sort_index().tolist())),
            " / ".join(map(str, dfc["Cluster_Jerarquico"].value_counts().sort_index().tolist())),
            " / ".join(map(str, dfc.loc[m_db, "Cluster_DBSCAN"].value_counts().sort_index().tolist())),
        ],
        "Ruido": ["Ninguno", "Ninguno", f"{(~m_db).sum()} ({100 * (~m_db).mean():.1f} %)"],
        "Silueta": [silhouette_score(X, dfc["Cluster_KMeans"]),
                    silhouette_score(X, dfc["Cluster_Jerarquico"]),
                    silhouette_score(X_arr[m_db], dfc.loc[m_db, "Cluster_DBSCAN"])],
    })
    res["ari"] = adjusted_rand_score(dfc["Cluster_KMeans"], dfc["Cluster_Jerarquico"])
    res["n_ruido"] = int((~m_db).sum())

    # Sensibilidad DBSCAN (Punto 13)
    filas = []
    for eps in [1.8, 2.0, 2.2, 2.5]:
        for ms in [5, 10, 15, 20]:
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_arr)
            mask = lab != -1
            n_cl = len(set(lab[mask]))
            filas.append({"eps": eps, "min_samples": ms, "Grupos": n_cl,
                          "Ruido (%)": round(100 * (~mask).mean(), 1),
                          "Silueta sin ruido": round(silhouette_score(X_arr[mask], lab[mask]), 3) if n_cl >= 2 else None})
    res["sensibilidad"] = pd.DataFrame(filas)

    # Punto 15: PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    dfc["PCA1"], dfc["PCA2"] = coords[:, 0], coords[:, 1]
    res["pca_var"] = pca.explained_variance_ratio_

    # Punto 16: nombres según la característica principal de cada grupo
    perfil_num = dfc.groupby("Cluster_KMeans")[VARS_PERFIL].mean()
    id_alto = int(perfil_num["Valor promedio"].idxmax())
    id_promo = int(perfil_num["Uso cupones en porcentaje"].idxmax())
    id_ocas = [int(c) for c in perfil_num.index if c not in (id_alto, id_promo)][0]
    nombres = {id_alto: "Alto valor por compra", id_promo: "Frecuentes sensibles a promociones",
               id_ocas: "Ocasionales de baja interacción"}
    dfc["Segmento"] = dfc["Cluster_KMeans"].map(nombres)
    dfc.insert(0, "Id cliente", df.loc[dfc.index, "Id cliente"].values)
    res["nombres"] = nombres

    # Punto 17: 3 clientes más cercanos a cada centroide
    reps = []
    for c in range(K):
        pos = np.where(dfc["Cluster_KMeans"].values == c)[0]
        dist = np.linalg.norm(X_arr[pos] - kmeans.cluster_centers_[c], axis=1)
        for j in np.argsort(dist)[:3]:
            fila = dfc.iloc[pos[j]]
            reps.append({"Segmento": nombres[c], "Id cliente": fila["Id cliente"],
                         **{v: fila[v] for v in VARS_PRINCIPALES + ["Uso cupones en porcentaje", "Días desde última compra"]},
                         "Distancia al centroide": round(float(dist[j]), 2)})
    res["representativos"] = pd.DataFrame(reps)

    res["df_clean"] = dfc
    res["X"] = X_arr
    res["modelo"] = {"kmeans": kmeans, "scaler": scaler, "dummies": list(X_cat.columns)}
    return res


def asignar_cliente(datos: dict, modelo: dict) -> int:
    """Punto 18: asigna un cliente nuevo con el mismo escalador, codificación y K-Means."""
    nuevo = pd.DataFrame([datos])
    num = pd.DataFrame(modelo["scaler"].transform(nuevo[FEATURES]), columns=FEATURES)
    cat = pd.get_dummies(nuevo[CATEGORICAS]).reindex(columns=modelo["dummies"], fill_value=0)
    X_nuevo = pd.concat([num, cat.reset_index(drop=True)], axis=1)
    return int(modelo["kmeans"].predict(X_nuevo)[0])


# ---------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------
st.sidebar.title("📊 Segmentación de clientes")
st.sidebar.caption("Caso 5 · Minería de Datos · Métodos no supervisados")
archivo = st.sidebar.file_uploader("Usar otro CSV (opcional)", type="csv",
                                   help="Debe tener las mismas 15 columnas del diccionario de datos.")
if archivo is not None:
    datos = pd.read_csv(archivo)
    st.sidebar.warning("Datos nuevos: K, eps y min_samples siguen fijos; revise codo, silueta y sensibilidad.")
elif CSV_LOCAL.exists():
    datos = pd.read_csv(CSV_LOCAL)
else:
    st.error("No se encontró clientes_ecommerce.csv junto a app.py. Súbalo desde la barra lateral.")
    st.stop()

R = procesar(datos)
dfc, nombres = R["df_clean"], R["nombres"]
orden_seg = [nombres[k] for k in sorted(nombres)]

pagina = st.sidebar.radio("Sección", [
    "Resumen", "Exploración de datos", "Modelos y comparación", "Segmentos", "Clasificar cliente nuevo"])

# ---------------------------------------------------------------------
# 1. Resumen
# ---------------------------------------------------------------------
if pagina == "Resumen":
    st.title("Segmentación de clientes de comercio electrónico")
    st.write("Se identificaron **3 segmentos** de clientes con K-Means, a partir de su comportamiento de compra, "
             "interacción con la plataforma y uso de promociones.")
    km = R["comparativa"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes analizados", f"{len(dfc):,}".replace(",", "."))
    c2.metric("Segmentos", K)
    c3.metric("Silueta K-Means", f"{km['Silueta']:.3f}")
    c4.metric("Coincidencia K-Means vs Jerárquico (ARI)", f"{R['ari']:.2f}")

    tam = dfc["Segmento"].value_counts().reindex(orden_seg)
    cols = st.columns(3)
    for col, seg in zip(cols, orden_seg):
        d = dfc[dfc["Segmento"] == seg]
        with col.container(border=True):
            st.subheader(seg)
            st.metric("Clientes", f"{tam[seg]} ({100 * tam[seg] / len(dfc):.1f} %)")
            st.write(f"Gasto total promedio: **USD {d['Gasto total últimos 6 meses'].mean():,.2f}**")
            st.write(f"Valor por compra: **USD {d['Valor promedio'].mean():,.2f}**")
            st.write(f"Compras en 6 meses: **{d['Compras últimos 6 meses'].mean():.1f}**")
            st.write(f"Uso de cupones: **{d['Uso cupones en porcentaje'].mean():.1f} %**")

    fig = px.pie(names=tam.index, values=tam.values, hole=0.5, color=tam.index, color_discrete_map=COLORES, category_orders={"color": orden_seg},
                 title="Distribución de clientes por segmento")
    st.plotly_chart(fig, width="stretch")
    st.caption("Los nombres describen comportamiento de compra; no implican que un segmento sea mejor que otro.")

# ---------------------------------------------------------------------
# 2. Exploración
# ---------------------------------------------------------------------
elif pagina == "Exploración de datos":
    st.title("Exploración y preparación de datos")
    df0 = R["df_original"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros originales", len(df0))
    c2.metric("Columnas", df0.shape[1])
    c3.metric("Registros después de limpieza", len(dfc))

    st.subheader("Distribución de las variables principales")
    var = st.selectbox("Variable", VARS_PRINCIPALES)
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df0, x=var, nbins=50, title=f"Histograma · {var}"), width="stretch")
    b.plotly_chart(px.box(df0, x=var, points="outliers", title=f"Valores atípicos · {var}"), width="stretch")
    st.dataframe(df0[VARS_PRINCIPALES].describe().round(2), width="stretch")

    st.subheader("Tratamiento de datos")
    st.table(pd.DataFrame(R["limpieza"].items(), columns=["Problema", "Registros"]))
    st.info(f"**Frecuencia mensual** se excluyó del modelo: su correlación con *Compras últimos 6 meses* es "
            f"{R['corr_frecuencia']:.4f} (equivale a compras ÷ 6). Se conserva solo para describir los segmentos.")

# ---------------------------------------------------------------------
# 3. Modelos
# ---------------------------------------------------------------------
elif pagina == "Modelos y comparación":
    st.title("Modelos de clustering y comparación")
    st.subheader("Selección del número de grupos (K)")
    a, b = st.columns(2)
    a.plotly_chart(px.line(R["codo"], x="K", y="Inercia", markers=True, title="Método del codo"), width="stretch")
    b.plotly_chart(px.line(R["codo"], x="K", y="Silueta", markers=True, title="Coeficiente de silueta"), width="stretch")

    st.subheader("Tabla comparativa")
    st.dataframe(R["comparativa"].style.format({"Silueta": "{:.4f}"}), width="stretch", hide_index=True)
    st.caption(f"Silueta de DBSCAN calculada sin ruido. ARI entre K-Means y Jerárquico: {R['ari']:.4f}.")

    st.subheader("Dendrograma (enlace Ward)")
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linkage(R["X"], method="ward"), truncate_mode="lastp", p=12, leaf_rotation=45, ax=ax)
    ax.axhline(y=50, color="red", linestyle="--", label="Corte en K=3")
    ax.set_ylabel("Distancia (Ward)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sensibilidad de DBSCAN")
    sens = R["sensibilidad"]
    st.plotly_chart(px.bar(sens, x="min_samples", y="Ruido (%)", color=sens["eps"].astype(str), barmode="group",
                           labels={"color": "eps"}, title="Porcentaje de ruido según eps y min_samples"),
                    width="stretch")
    st.dataframe(sens, width="stretch", hide_index=True)
    st.caption(f"Configuración elegida: eps = {EPS}, min_samples = {MIN_SAMPLES}.")

# ---------------------------------------------------------------------
# 4. Segmentos
# ---------------------------------------------------------------------
elif pagina == "Segmentos":
    st.title("Perfil de los segmentos")
    v = R["pca_var"]
    fig = px.scatter(dfc, x="PCA1", y="PCA2", color="Segmento", color_discrete_map=COLORES, category_orders={"Segmento": orden_seg},
                     hover_data=["Id cliente", "Gasto total últimos 6 meses", "Compras últimos 6 meses"],
                     title=f"Proyección PCA (explica el {100 * v.sum():.1f} % de la variación)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Perfil promedio")
    perfil = dfc.groupby("Segmento")[VARS_PERFIL].mean().reindex(orden_seg).round(2)
    st.dataframe(perfil.T, width="stretch")

    var = st.selectbox("Comparar segmentos por", VARS_PERFIL, index=2)
    st.plotly_chart(px.bar(perfil.reset_index(), x="Segmento", y=var, color="Segmento",
                           color_discrete_map=COLORES, title=f"{var} promedio por segmento"), width="stretch")

    st.subheader("Clientes representativos (3 más cercanos al centroide)")
    st.dataframe(R["representativos"], width="stretch", hide_index=True)

    st.subheader("Buscar clientes")
    seg = st.multiselect("Segmento", orden_seg, default=orden_seg)
    st.dataframe(dfc[dfc["Segmento"].isin(seg)].drop(columns=["Cluster_Jerarquico", "Cluster_DBSCAN", "PCA1", "PCA2"]),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------------
# 5. Clasificar cliente nuevo
# ---------------------------------------------------------------------
else:
    st.title("Prueba de concepto: clasificar un cliente nuevo")
    st.write("Ingrese los datos del cliente. Se aplica el mismo escalado, codificación y modelo K-Means del análisis.")
    med = dfc[FEATURES].median()
    with st.form("cliente"):
        cols = st.columns(3)
        datos_cliente = {}
        for i, f in enumerate(FEATURES):
            datos_cliente[f] = cols[i % 3].number_input(f, min_value=0.0, value=float(round(med[f], 2)))
        a, b = st.columns(2)
        datos_cliente["Categoría más frecuente"] = a.selectbox(
            "Categoría más frecuente", sorted(dfc["Categoría más frecuente"].unique()))
        datos_cliente["Canal preferido"] = b.selectbox("Canal preferido", sorted(dfc["Canal preferido"].unique()))
        enviar = st.form_submit_button("Asignar segmento", type="primary")

    if enviar:
        grupo = asignar_cliente(datos_cliente, R["modelo"])
        st.success(f"El cliente fue asignado al segmento: **{nombres[grupo]}** (clúster {grupo})")
        perfil = dfc[dfc["Cluster_KMeans"] == grupo][VARS_PRINCIPALES + ["Uso cupones en porcentaje"]].mean()
        comp = pd.DataFrame({"Cliente ingresado": pd.Series(datos_cliente)[perfil.index].astype(float),
                             "Promedio del segmento": perfil.round(2)})
        st.dataframe(comp, width="stretch")

    with st.expander("Validación del modelo"):
        muestra = dfc.head(100)
        ok = sum(asignar_cliente(f[FEATURES + CATEGORICAS].to_dict(), R["modelo"]) == f["Cluster_KMeans"]
                 for _, f in muestra.iterrows())
        st.write(f"Coincidencia en 100 clientes del entrenamiento: **{ok}/100**")

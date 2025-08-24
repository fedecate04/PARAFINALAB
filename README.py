# app.py
# -*- coding: utf-8 -*-
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd

# --- Corrección: fijar backend y evitar instalación en caliente ---
import matplotlib
matplotlib.use("Agg")  # backend headless para servidores/Streamlit
import matplotlib.pyplot as plt
# -----------------------------------------------------------------

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ==============================
# App metadata y estilo
# ==============================
PAGE_TITLE = "FLUIDOS MULTIFÁSICOS EN LA INDUSTRIA DEL PETRÓLEO"
PAGE_SUBTITLE = "CRISTALIZACIÓN Y DEPOSICIÓN DE PARAFINAS"
PROFESOR = "Ing. Ezequiel Krumrick"
ALUMNO = "Catereniuc Federico"

st.set_page_config(
    page_title="WAT – Parafinas | Streamlit",
    page_icon="🧪",
    layout="wide"
)

# Paleta sencilla
PRIMARY = "#0F766E"   # teal-700
ACCENT  = "#0891B2"   # cyan-600
MUTED   = "#475569"   # slate-600

# ==============================
# Utilidades
# ==============================
def read_csv_no_header(file) -> pd.DataFrame:
    """
    Lee CSV sin encabezado: col0 = Temperatura [°C], col1 = Viscosidad [Pa·s]
    Limpia NaN, no positivos y ordena por T ascendente.
    """
    df = pd.read_csv(file, header=None)
    df = df.rename(columns={0: "T_C", 1: "mu_Pa_s"})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["mu_Pa_s"] > 0)]
    df = df.sort_values("T_C").reset_index(drop=True)
    return df

def clean_manual_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["T_C", "mu_Pa_s"]
    df["T_C"] = pd.to_numeric(df["T_C"], errors="coerce")
    df["mu_Pa_s"] = pd.to_numeric(df["mu_Pa_s"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["mu_Pa_s"] > 0)]
    df = df.sort_values("T_C").reset_index(drop=True)
    return df

def segmented_two_lines(x, y, min_pts=3):
    """
    Búsqueda de quiebre (índice k) que minimiza SSE global de dos regresiones lineales.
    x: 1/T [K^-1], y: log10(mu)
    Retorna dict con: k_opt, coefs1, coefs2, sse_min, x_break, y_break.
    """
    n = len(x)
    if n < 2*min_pts:
        return None

    best = {
        "k_opt": None,
        "coefs1": None,
        "coefs2": None,
        "sse_min": np.inf,
        "x_break": None,
        "y_break": None
    }

    for k in range(min_pts-1, n - min_pts):
        x1, y1 = x[:k+1], y[:k+1]
        x2, y2 = x[k+1:], y[k+1:]
        m1, b1 = np.polyfit(x1, y1, 1)
        m2, b2 = np.polyfit(x2, y2, 1)
        sse1 = np.sum((y1 - (m1*x1 + b1))**2)
        sse2 = np.sum((y2 - (m2*x2 + b2))**2)
        sse = sse1 + sse2
        if sse < best["sse_min"]:
            if abs(m1 - m2) > 1e-12:
                x_int = (b2 - b1) / (m1 - m2)
                y_int = m1*x_int + b1
            else:
                x_int = (x[k] + x[k+1]) / 2.0
                y_int = (y[k] + y[k+1]) / 2.0

            best = {
                "k_opt": k,
                "coefs1": (m1, b1),
                "coefs2": (m2, b2),
                "sse_min": sse,
                "x_break": x_int,
                "y_break": y_int
            }

    return best

def x_to_TC(x_invK):
    """1/T[K] -> T[°C]"""
    T_K = 1.0 / x_invK
    return T_K - 273.15

def prepare_xy(df: pd.DataFrame):
    """
    Prepara x=1/T[K], y=log10(mu) desde T[°C], mu[Pa·s].
    """
    T_K = df["T_C"].values + 273.15
    x = 1.0 / T_K
    y = np.log10(df["mu_Pa_s"].values)
    return x, y

def make_figure(df: pd.DataFrame, fit, estilo="apunte"):
    """
    Figura estilo 'apunte':
      - y: μ [Pa·s] en escala log
      - x: T [°C]
    El ajuste se hace en (log10 μ) vs (1/T) y se transforma a μ vs T para visualizar.
    """
    # Datos
    T_C = df["T_C"].values
    mu   = df["mu_Pa_s"].values

    # Coeficientes del ajuste en espacio (x=1/T[K], y=log10(mu))
    m1, b1 = fit["coefs1"]
    m2, b2 = fit["coefs2"]
    k      = fit["k_opt"]
    xb, yb = fit["x_break"], fit["y_break"]   # xb = (1/T[K])*, yb = log10(mu)*

    # Ejes: T [°C] como x principal
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Dispersión de datos en μ vs T
    ax.scatter(T_C, mu, color=PRIMARY, label="Datos", zorder=3)

    # Construir curvas de ajuste EN μ vs T:
    T1 = T_C[:k+1]
    T2 = T_C[k+1:]

    def mu_fit_from_line(Tc_array, m, b):
        Tk = Tc_array + 273.15
        x_invK = 1.0 / Tk
        y_log10 = m * x_invK + b
        return np.power(10.0, y_log10)

    # Curvas suaves por tramo
    T1_line = np.linspace(T1.min(), T1.max(), 80) if len(T1) >= 2 else T1
    T2_line = np.linspace(T2.min(), T2.max(), 80) if len(T2) >= 2 else T2

    mu1_line = mu_fit_from_line(T1_line, m1, b1)
    mu2_line = mu_fit_from_line(T2_line, m2, b2)

    ax.plot(T1_line, mu1_line, color=ACCENT, lw=2, label="Ajuste tramo 1")
    ax.plot(T2_line, mu2_line, color=MUTED,  lw=2, label="Ajuste tramo 2")

    # Punto crítico (WAT) en coordenadas T[°C]–μ
    T_wat = x_to_TC(xb)
    mu_wat = np.power(10.0, yb)
    ax.scatter([T_wat], [mu_wat], color="red", s=70, zorder=4, label="Punto crítico (WAT)")

    # Estilo de ejes (log en Y)
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
    ax.set_yscale("log", base=10)
    # rango sugerido; podés comentar si querés autoscale
    ymin = min(1e-3, np.nanmin(mu)*0.8) if np.all(np.isfinite(mu)) else 1e-3
    ymax = max(1e2,  np.nanmax(mu)*1.2) if np.all(np.isfinite(mu)) else 1e2
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("Temperatura, T [°C]")
    ax.set_ylabel("Viscosidad, μ [Pa·s]")
    ax.grid(alpha=0.25, which="both", axis="both")
    ax.legend(frameon=False, loc="best")
    ax.set_title(f"Determinación de WAT   {T_wat:.1f} °C")

    # Eje superior opcional con 1/T [K^-1]
    ax_top = ax.secondary_xaxis(
        'top',
        functions=(lambda Tc: 1.0/(Tc+273.15), lambda invT: (1.0/invT)-273.15)
    )
    xticks = np.linspace(T_C.min(), T_C.max(), 6) if len(T_C) >= 2 else T_C
    ax.set_xticks(xticks)
    ax_top.set_xticks(xticks)
    ax_top.set_xlabel(r"$1/T$  [K$^{-1}$]")

    # Anotación WAT
    ax.annotate(f"WAT ≈ {T_wat:.2f} °C",
                xy=(T_wat, mu_wat),
                xytext=(T_wat + 0.03*(T_C.max()-T_C.min()), mu_wat*1.3),
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red")

    plt.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def fig_to_pdf_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def maybe_load_logo():
    """
    Carga logoutn.png desde el directorio del app (repo local). Si no existe, devuelve None.
    """
    path = "logoutn.png"
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None
    return None

def export_pedagogical_pdf(df, fit, fig_png_bytes, meta, custom_logo_bytes=None):
    """
    PDF pedagógico (A4) con portada, datos, figura y explicación.
    Maquetación corregida para evitar superposición de logo y títulos.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    W, H = A4

    # ===== Portada / Encabezado =====
    margin_x = 1.5*cm
    margin_y = 1.5*cm
    header_height = 3.0*cm  # banda de encabezado

    # Banda superior
    c.setFillColorRGB(0.93, 0.99, 1.00)  # celeste muy claro
    c.rect(0, H - header_height, W, header_height, stroke=0, fill=1)

    # Logo a la derecha
    if custom_logo_bytes:
        try:
            logo = ImageReader(io.BytesIO(custom_logo_bytes))
            logo_w = 3.8*cm
            logo_h = 2.2*cm
            c.drawImage(
                logo,
                x=W - margin_x - logo_w,
                y=H - margin_y - logo_h,
                width=logo_w,
                height=logo_h,
                mask='auto',
                preserveAspectRatio=True,
                anchor='sw'
            )
        except Exception:
            pass

    # Título a la izquierda
    c.setFillColorRGB(0.06, 0.46, 0.43)  # PRIMARY
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_x, H - margin_y - 0.8*cm, meta["page_title"])
    c.setFillColorRGB(0.03, 0.57, 0.70)  # ACCENT
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin_x, H - margin_y - 1.5*cm, meta["page_subtitle"])

    # Línea divisoria
    c.setStrokeColorRGB(0.75, 0.90, 0.95)
    c.setLineWidth(1.2)
    c.line(margin_x, H - header_height - 0.2*cm, W - margin_x, H - header_height - 0.2*cm)

    # Metadatos (debajo del encabezado)
    y = H - header_height - 0.8*cm
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 11)
    c.drawString(margin_x, y, f"Profesor: {meta['profesor']}")
    y -= 0.6*cm
    c.drawString(margin_x, y, f"Alumno: {meta['alumno']}")
    y -= 0.6*cm
    c.drawString(margin_x, y, f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Resultado WAT
    xb = fit["x_break"]
    T_wat = x_to_TC(xb)
    y -= 1.0*cm
    c.setFont("Helvetica-Bold", 12)
    c.setFillColorRGB(0.06, 0.46, 0.43)
    c.drawString(margin_x, y, f"Resultado principal → WAT ≈ {T_wat:.2f} °C")
    c.setFillColorRGB(0,0,0)

    # Figura
    y -= 0.5*cm
    try:
        img = ImageReader(io.BytesIO(fig_png_bytes))
        fig_w = W - 2*margin_x
        fig_h = fig_w * 0.62
        # si no entra, ajusta
        if y - fig_h < margin_y:
            fig_h = y - margin_y
            fig_w = fig_h / 0.62
        c.drawImage(img, x=margin_x, y=y - fig_h, width=fig_w, height=fig_h, mask='auto')
        y = y - fig_h - 0.8*cm
    except Exception:
        y -= 0.5*cm

    # ===== Página 2: explicación =====
    c.showPage()

    margin = 2.0*cm
    y = H - margin

    def write_paragraph(text, leading=14, font="Helvetica", size=11):
        nonlocal y
        c.setFont(font, size)
        width_chars = 95  # envoltura simple
        import textwrap
        for line in textwrap.wrap(text, width=width_chars):
            c.drawString(margin, y, line)
            y -= leading

    c.setFillColorRGB(0.06, 0.46, 0.43)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Interpretación y fundamentos")
    y -= 18
    c.setFillColorRGB(0,0,0)

    x_arr, ylog = prepare_xy(df)
    k = fit["k_opt"]
    m1, b1 = fit["coefs1"]
    m2, b2 = fit["coefs2"]

    write_paragraph(
        "El método estima la Temperatura de Aparición de Cera (WAT) detectando un cambio "
        "de pendiente en la curva log10(μ) vs 1/T[K]. Sobre el tramo de altas temperaturas, la "
        "viscosidad obedece un comportamiento Arrhenius casi lineal; al enfriar por debajo del "
        "WAT se intensifican los efectos de cristalización y la pendiente efectiva cambia."
    )
    write_paragraph(
        f"En el conjunto de datos analizado (N={len(df)}), la partición óptima se halló en k={k}, "
        "minimizando el error cuadrático total de dos regresiones lineales. La intersección de "
        "las rectas define el punto crítico desde el cual se informa WAT."
    )
    write_paragraph(
        f"Resultado: WAT ≈ {T_wat:.2f} °C. Sugerencia operativa (regla simple): mantener "
        "T_pared > WAT + 3 °C para reducir fuertemente la tasa de depósito."
    )

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Glosario mínimo")
    y -= 16
    c.setFont("Helvetica", 11)
    write_paragraph("WAT: Temperatura a la que aparece la primera fase sólida (parafina) en equilibrio.")
    write_paragraph("WDT: Temperatura de desaparición de cera al recalentar (típicamente WAT + 2–5 °C).")
    write_paragraph("Tixotropía: Pérdida de estructura gel al someter a cizalla y recuperación en reposo.")
    write_paragraph("Modelo HB: Herschel–Bulkley con esfuerzo de fluencia τ0 y exponente n.")

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Buenas prácticas de ingeniería")
    y -= 16
    c.setFont("Helvetica", 11)
    write_paragraph("1) Verificar sensibilidad a puntos atípicos (outliers) y repetir el ajuste.")
    write_paragraph("2) Confirmar WAT con técnica complementaria (p. ej., DSC o turbidez ASTM).")
    write_paragraph("3) Documentar hipótesis (limpieza de datos, mínimos puntos por tramo).")
    write_paragraph("4) Registrar fecha, lote de crudo y condiciones de medición (trazabilidad).")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def download_button_bytes(data_bytes, filename, label):
    st.download_button(
        label=label,
        data=data_bytes,
        file_name=filename,
        mime="application/octet-stream"
    )

# ==============================
# Estado de sesión
# ==============================
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["T_C", "mu_Pa_s"])
if "fit" not in st.session_state:
    st.session_state.fit = None
if "fig_png" not in st.session_state:
    st.session_state.fig_png = None
if "fig_pdf" not in st.session_state:
    st.session_state.fig_pdf = None
if "logo_bytes" not in st.session_state:
    st.session_state.logo_bytes = maybe_load_logo()

# ==============================
# Sidebar navegación
# ==============================
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", ["Principal", "Cargar datos", "Ajuste WAT", "Teoría", "Acerca de"])

st.sidebar.markdown("---")
st.sidebar.caption("Asignatura electiva · UTN – FRN")

# ==============================
# Encabezado visual
# ==============================
def header():
    col_logo, col_title = st.columns([1, 5], vertical_alignment="center")
    with col_logo:
        # MOSTRAR SIEMPRE EL LOGO DEL REPO (logoutn.png) SI EXISTE
        if st.session_state.logo_bytes:
            st.image(st.session_state.logo_bytes, use_container_width=True)
    with col_title:
        st.markdown(
            f"""
            <div style="padding:6px 12px;border-radius:12px;background:#ecfeff;border:1px solid #a5f3fc">
              <h1 style="margin:0;color:{PRIMARY};font-weight:800;">{PAGE_TITLE}</h1>
              <h3 style="margin:4px 0 0 0;color:{ACCENT};font-weight:700;">{PAGE_SUBTITLE}</h3>
              <p style="margin:6px 0 0 0;color:{MUTED};">
                <strong>PROFESOR:</strong> {PROFESOR} &nbsp; | &nbsp; <strong>ALUMNO:</strong> {ALUMNO}
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================
# Páginas
# ==============================
if page == "Principal":
    header()
    st.markdown(
        """
        ### Objetivo de la app
        Estimar la **Temperatura de Aparición de Cera (WAT)** a partir de datos de **Temperatura [°C]** y 
        **Viscosidad [Pa·s]** mediante ajuste por **pendiente quebrada** en la curva **log10(μ) vs 1/T[K]**.  
        La app:
        - Limpia y valida datos,  
        - Busca el **punto de quiebre** que minimiza el error global de dos rectas,  
        - Calcula y reporta **WAT**,  
        - Genera una **figura** con los ajustes y el punto crítico,  
        - Exporta imagen **PNG/PDF** y un **PDF pedagógico** con explicación técnica.
        """
    )
    st.info("Sugerencia: Iniciá en **Cargar datos** y luego pasá a **Ajuste WAT**.")

elif page == "Cargar datos":
    header()
    st.subheader("1) Subir CSV (sin encabezado) o ingresar manualmente")
    tab_csv, tab_manual = st.tabs(["📥 Subir CSV", "⌨️ Ingreso manual"])

    with tab_csv:
        file = st.file_uploader("CSV sin encabezado (col1=T[°C], col2=μ[Pa·s])", type=["csv"])
        if file is not None:
            try:
                df = read_csv_no_header(file)
                st.session_state.data = df
                st.success(f"Datos cargados: {len(df)} filas válidas.")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error al leer CSV: {e}")

    with tab_manual:
        st.markdown("Ingresá los pares (T[°C], μ[Pa·s]). Podés editar la tabla libremente.")
        if st.session_state.data.empty:
            base = pd.DataFrame({"T_C":[25.0, 30.0, 35.0], "mu_Pa_s":[0.15, 0.08, 0.05]})
        else:
            base = st.session_state.data.copy()

        edited = st.data_editor(
            base,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_editor",
            column_config={
                "T_C": st.column_config.NumberColumn("T [°C]", step=0.1, format="%.3f"),
                "mu_Pa_s": st.column_config.NumberColumn("μ [Pa·s]", step=1e-4, format="%.6f")
            }
        )
        if st.button("Guardar tabla", type="primary"):
            df = clean_manual_df(pd.DataFrame(edited))
            st.session_state.data = df
            st.success(f"Tabla guardada ({len(df)} filas válidas).")

elif page == "Ajuste WAT":
    header()
    st.subheader("2) Ajuste por pendiente quebrada y exportación")

    df = st.session_state.data
    if df.empty or len(df) < 6:
        st.warning("Cargá al menos 6 datos válidos en la sección **Cargar datos**.")
    else:
        st.write("**Datos vigentes:**")
        st.dataframe(df, use_container_width=True, height=240)

        min_pts = st.slider("Mínimo de puntos por tramo", min_value=3, max_value=10, value=3, step=1,
                            help="A mayor mínimo, más robusto el ajuste por tramo.")
        x, y = prepare_xy(df)
        fit = segmented_two_lines(x, y, min_pts=min_pts)

        if not fit:
            st.error("No se pudo realizar el ajuste. Verificá la cantidad/calidad de datos.")
        else:
            st.session_state.fit = fit
            T_wat = x_to_TC(fit["x_break"])

            colA, colB = st.columns([2, 1])
            with colA:
                fig = make_figure(df, fit)
                st.pyplot(fig, use_container_width=True)

                fig_png = fig_to_png_bytes(fig)
                fig_pdf = fig_to_pdf_bytes(fig)
                st.session_state.fig_png = fig_png
                st.session_state.fig_pdf = fig_pdf

            with colB:
                k = fit["k_opt"]
                m1, b1 = fit["coefs1"]
                m2, b2 = fit["coefs2"]
                st.metric("WAT estimada [°C]", f"{T_wat:.2f}")
                st.caption(f"Índice de quiebre k={k} (partición de datos).")

                st.markdown("**Parámetros de ajuste**")
                st.code(
f"""Tramo 1: y = m1*x + b1
m1 = {m1:.5f}, b1 = {b1:.5f}

Tramo 2: y = m2*x + b2
m2 = {m2:.5f}, b2 = {b2:.5f}
""",
                    language="text"
                )

                st.markdown("**Descargas**")
                colD1, colD2 = st.columns(2)
                with colD1:
                    download_button_bytes(fig_png, "WAT_figure.png", "⬇️ Figura PNG (300 dpi)")
                with colD2:
                    st.download_button(
                        label="⬇️ Figura PDF",
                        data=fig_pdf,
                        file_name="WAT_figure.pdf",
                        mime="application/pdf"
                    )

                # Reporte pedagógico
                if st.button("📄 Generar Reporte Pedagógico (PDF)", type="primary"):
                    meta = {
                        "page_title": PAGE_TITLE,
                        "page_subtitle": PAGE_SUBTITLE,
                        "profesor": PROFESOR,
                        "alumno": ALUMNO
                    }
                    pdf_bytes = export_pedagogical_pdf(
                        df=df,
                        fit=fit,
                        fig_png_bytes=fig_png,
                        meta=meta,
                        custom_logo_bytes=st.session_state.logo_bytes
                    )
                    st.download_button(
                        "⬇️ Descargar Reporte PDF",
                        data=pdf_bytes,
                        file_name="Reporte_WAT_pedagógico.pdf",
                        mime="application/pdf"
                    )

            st.info(
                "Tip: verificá la sensibilidad de la WAT removiendo outliers, cambiando el mínimo de puntos "
                "por tramo o comparando con otro dataset."
            )

elif page == "Teoría":
    header()
    st.subheader("3) Ventana teórica (resumen guiado)")
    with st.expander("📌 ¿Qué es el WAT?"):
        st.write(
            "La **Temperatura de Aparición de Cera (WAT)** es la temperatura a la que precipita la primera "
            "fase sólida de parafinas en equilibrio. Operar por debajo del WAT incrementa la viscosidad "
            "y favorece la formación de depósitos en pared."
        )
    with st.expander("📉 Método de pendiente quebrada"):
        st.write(
            "Si graficamos **log10(μ)** contra **1/T[K]**, muchos crudos muestran un tramo casi lineal a "
            "altas temperaturas (comportamiento Arrhenius). Al enfriar, la pendiente efectiva cambia por "
            "cristalización. El **punto de quiebre** entre dos rectas ajustadas aproxima el WAT."
        )
    with st.expander("🧪 Buenas prácticas de laboratorio/datos"):
        st.markdown(
            "- Medí **μ** en un rango suficiente de T.\n"
            "- Eliminá **μ ≤ 0** y valores no confiables.\n"
            "- Confirmá con técnica complementaria (p. ej. turbidez o DSC).\n"
            "- Documentá fecha, lote y condiciones (trazabilidad)."
        )
    with st.expander("⚙️ Ingeniería y operación"):
        st.markdown(
            "- Regla práctica: mantener **T_pared > WAT + 3 °C** para atenuar depósitos.\n"
            "- Complementar con **inhibidores**, **aislamiento térmico** y **pigging**.\n"
            "- Estimar la **VMT** (Velocidad Mínima de Transporte) al diseñar/operar ductos."
        )
    st.success("RA1–CE1 → entendimiento fenomenológico y lectura crítica de curvas μ–T.")

elif page == "Acerca de":
    header()
    st.markdown(
        """
        **Versión:** 1.1  
        **Función:** Detección de WAT y creación de reportes.  
        **Créditos:** Cátedra Flujos Multifásicos – UTN FRN.  
        **Licencia:** Uso académico.
        """
    )

# Footer sutil
st.markdown(
    f"""
    <hr style="opacity:.15"/>
    <div style="color:#64748b;font-size:12px">
    © {datetime.now().year} · UTN – FR Neuquén · App educativa para estimación de WAT por pendiente quebrada.
    </div>
    """,
    unsafe_allow_html=True
)

"""
Streamlit app — KPIs de Branqueamento
Histogramas, Capabilidade e Filtros Dinâmicos
Rev. 13-Aug-2025 — legenda PT-BR configurável + Nome(Tag) + figuras Nome_Tag
  • Cp/Cpk com desvio-padrão GLOBAL (Std).
  • Pp/Ppk com σ within (MR/D2) apenas como referência.
  • Limites teóricos (±3·σ_within) opcionais (visual/uso comparativo).
  • Legenda configurável (Seção 3.1), tudo marcado por padrão, em português.
  • Filtros/seleção mostram "Nome (Tag)". Figuras salvas como "Nome_Tag.png".
  • Produção = filtro GLOBAL. Demais variáveis = filtros LOCAIS por gráfico.
  • Remoção de outliers por variável (IQR ou Z-score).
  • Regra adicional: após Produção, cada variável do histograma é forçada a > 0.
"""

import io
import re
import zipfile
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import streamlit as st

# ========== Aparência global ==========
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

st.set_page_config(page_title="KPIs Branqueamento — Histogramas + Capabilidade", layout="wide")
st.title("KPIs de Branqueamento — Análise Rápida (Histogramas + Capabilidade)")

# ========= Utilitários =========
@st.cache_data(show_spinner=False)
def read_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def coerce_numeric_columns(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def safe_cv(m: float, s: float):
    return np.nan if (m is None or np.isnan(m) or abs(m) < 1e-12) else 100 * s / m

# d2 para MR de 2 pontos (Individuals/MR)
D2 = 1.128

def sigma_within_from_mr(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.size < 2:
        return np.nan
    return x.diff().abs().dropna().mean() / D2

def slug(s: str) -> str:
    """Sanitiza para nomes de arquivo: 'Nome da Variável (TAG)' -> 'Nome_da_Variavel_TAG'."""
    s = re.sub(r"[^\w\s\-\.]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    return s

# ===== Outliers =====
def remove_outliers_series(s: pd.Series, method: str = "IQR", k: float = 1.5, zthr: float = 3.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return x
    if method == "IQR":
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        return x[(x >= lo) & (x <= hi)]
    elif method == "Z-score":
        mu, sd = x.mean(), x.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            return x
        z = (x - mu) / sd
        return x[np.abs(z) <= zthr]
    else:
        return x  # sem remoção

# ================== Capabilidade (σ GLOBAL para Cp/Cpk; σ within para Pp/Ppk) ==================
def capability_indices(s: pd.Series, lo: Optional[float], hi: Optional[float]):
    x = pd.to_numeric(s, errors="coerce").dropna()
    N, mu = int(x.size), x.mean()
    sd_overall = x.std(ddof=1)          # σ GLOBAL — Cp/Cpk
    sd_within = sigma_within_from_mr(x) # σ within — Pp/Ppk

    cap = {
        "N": N, "Mean": mu, "Std": sd_overall, "sd_within": sd_within,
        "Cp": np.nan, "Cpk": np.nan, "Pp": np.nan, "Ppk": np.nan,
        "Cpu": np.nan, "Cpl": np.nan, "Ppu": np.nan, "Ppl": np.nan,
        "Zbench": np.nan
    }

    if lo is not None and hi is not None and hi > lo:
        width = hi - lo
        if np.isfinite(sd_overall) and sd_overall > 0:
            cap["Cp"]  = width / (6 * sd_overall)
            cap["Cpk"] = min((hi - mu) / (3 * sd_overall), (mu - lo) / (3 * sd_overall))
        if np.isfinite(sd_within) and sd_within > 0:
            cap["Pp"]  = width / (6 * sd_within)
            cap["Ppk"] = min((hi - mu) / (3 * sd_within), (mu - lo) / (3 * sd_within))

    elif hi is not None:
        if np.isfinite(sd_overall) and sd_overall > 0:
            cap["Cpu"] = cap["Cpk"] = (hi - mu) / (3 * sd_overall)
        if np.isfinite(sd_within) and sd_within > 0:
            cap["Ppu"] = cap["Ppk"] = (hi - mu) / (3 * sd_within)

    elif lo is not None:
        if np.isfinite(sd_overall) and sd_overall > 0:
            cap["Cpl"] = cap["Cpk"] = (mu - lo) / (3 * sd_overall)
        if np.isfinite(sd_within) and sd_within > 0:
            cap["Ppl"] = cap["Ppk"] = (mu - lo) / (3 * sd_within)

    if np.isfinite(cap.get("Cpk", np.nan)):
        cap["Zbench"] = 3 * cap["Cpk"]

    return cap

# ================== Gráfico ==================
def gen_hist(
    s: pd.Series,
    title: str,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    bins: int = 40,
    pal: Optional[Dict[str, str]] = None,
    show_norm: bool = False,
    # ---- Limites teóricos (SPC 3σ within) ----
    use_theoretical: bool = False,
    theo_line_color: str = "#1F618D",
    theo_fill_color: str = "#D6EAF8",
    theo_fill_alpha: float = 0.18,
    use_theoretical_for_capability: bool = False,
    # ---- Legenda configurável ----
    parametros_legenda: Optional[List[str]] = None,
):
    """
    Histograma com:
      • LO/HI do catálogo (linhas tracejadas).
      • Faixa teórica (±3·σ_within) opcional.
      • Legenda lateral configurável (PT-BR).
    """
    if parametros_legenda is None:
        parametros_legenda = []

    colors = pal or {"bars": "#4B91CC", "lo": "#C0392B", "hi": "#6C3483"}

    s = pd.to_numeric(s, errors="coerce").dropna()
    media = s.mean()
    desvio = s.std(ddof=1)
    cv_pct = safe_cv(media, desvio)
    vmin = s.min() if s.size else np.nan
    vmax = s.max() if s.size else np.nan

    # % em relação a LO/HI
    pct_dentro = pct_abaixo = pct_acima = np.nan
    if lo is not None and hi is not None and hi > (lo if lo is not None else -np.inf):
        pct_dentro = 100.0 * ((s >= lo) & (s <= hi)).mean()
        pct_abaixo = 100.0 * (s < lo).mean()
        pct_acima  = 100.0 * (s > hi).mean()
    elif lo is not None:
        pct_abaixo = 100.0 * (s < lo).mean()
    elif hi is not None:
        pct_acima  = 100.0 * (s > hi).mean()

    # Limites teóricos (±3·σ_within) — VISUAL
    sigma_w = sigma_within_from_mr(s)
    theo_lcl = theo_ucl = np.nan
    if use_theoretical and np.isfinite(sigma_w) and sigma_w > 0:
        theo_lcl = media - desvio
        theo_ucl = media + desvio

    # % teórico
    pct_dentro_teo = pct_abaixo_teo = pct_acima_teo = np.nan
    if use_theoretical and np.isfinite(theo_lcl) and np.isfinite(theo_ucl) and theo_ucl > theo_lcl:
        pct_dentro_teo = 100.0 * ((s >= theo_lcl) & (s <= theo_ucl)).mean()
        pct_abaixo_teo = 100.0 * (s < theo_lcl).mean()
        pct_acima_teo  = 100.0 * (s > theo_ucl).mean()

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(10, 5.2))
    counts, bin_edges, _ = ax.hist(
        s, bins=bins, color=colors["bars"], edgecolor="white", linewidth=0.8, alpha=0.9
    )

    # LO/HI catálogo (tracejado)
    if lo is not None:
        ax.axvline(lo, color=colors["lo"], ls="--", lw=2)
    if hi is not None:
        ax.axvline(hi, color=colors["hi"], ls="--", lw=2)

    # Faixa + linhas teóricas (sólidas) — VISUAL
    if use_theoretical and np.isfinite(theo_lcl) and np.isfinite(theo_ucl) and theo_ucl > theo_lcl:
        ax.axvspan(theo_lcl, theo_ucl, color=theo_fill_color, alpha=theo_fill_alpha, zorder=0)
        ax.axvline(theo_lcl, color=theo_line_color, ls="-", lw=2)
        ax.axvline(theo_ucl, color=theo_line_color, ls="-", lw=2)

    # Curva normal opcional (usando std global)
    if show_norm and np.isfinite(desvio) and desvio > 0 and s.size > 0:
        xmin, xmax = ax.get_xlim()
        xs  = np.linspace(xmin, xmax, 300)
        pdf = (1.0 / (desvio * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - media) / desvio) ** 2)
        binw = (bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        ax.plot(xs, pdf * s.size * binw, lw=2)

    ax.set_title(title, weight="bold", pad=8)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frequência")
    ax.margins(x=0.02)

    # Legenda lateral — itens básicos (linhas e faixas)
    handles = [Patch(facecolor=colors["bars"], label=title)]
    if lo is not None:
        handles.append(Line2D([], [], color=colors["lo"], ls="--", lw=2, label=f"LO: {lo:g}"))
    if hi is not None:
        handles.append(Line2D([], [], color=colors["hi"], ls="--", lw=2, label=f"HI: {hi:g}"))
    if use_theoretical and np.isfinite(theo_lcl) and np.isfinite(theo_ucl):
        handles += [
            Line2D([], [], color=theo_line_color, ls="-", lw=2, label=f"LCL teórico: {theo_lcl:.3g}"),
            Line2D([], [], color=theo_line_color, ls="-", lw=2, label=f"UCL teórico: {theo_ucl:.3g}"),
            Patch(facecolor=theo_fill_color, alpha=theo_fill_alpha, label="Faixa ±3·σ (teórica)"),
        ]

    # ========= Helpers de formatação robustos =========
    def fmt_val(v, sig=3, suffix="", pct=False):
        """Formata v com sig figs (ou 2 casas se pct=True). Lida com numpy/pandas e NaN."""
        try:
            if hasattr(v, "item"):
                v = v.item()
            v = float(v)
        except Exception:
            return "N/A"
        if not np.isfinite(v):
            return "N/A"
        if pct:
            return f"{v:.2f}{suffix}"
        return f"{v:.{sig}g}{suffix}"

    def add_line(label_pt: str, value, pct=False):
        """Adiciona linha na legenda se o rótulo estiver selecionado em parametros_legenda."""
        if label_pt not in parametros_legenda:
            return
        txt = fmt_val(value, sig=3, suffix="%" if pct else "", pct=pct)
        handles.append(Patch(facecolor="none", edgecolor="none", label=f"{label_pt}: {txt}"))

    # ========= Legenda — parâmetros selecionáveis (PT-BR) =========
    add_line("Média",                    media)
    add_line("Desvio-padrão",            desvio)
    add_line("Coeficiente de variação",  cv_pct, pct=True)
    add_line("Mínimo",                   vmin)
    add_line("Máximo",                   vmax)
    add_line("% dentro",                 pct_dentro, pct=True)
    add_line("% abaixo",                 pct_abaixo, pct=True)
    add_line("% acima",                  pct_acima, pct=True)
    add_line("% dentro (teórico)",       pct_dentro_teo, pct=True)
    add_line("% abaixo (teórico)",       pct_abaixo_teo, pct=True)
    add_line("% acima (teórico)",        pct_acima_teo, pct=True)

    # ========= Capabilidade =========
    lo_cap, hi_cap = lo, hi
    cap_label = "Capabilidade (σ global)"
    if use_theoretical_for_capability and np.isfinite(theo_lcl) and np.isfinite(theo_ucl) and theo_ucl > theo_lcl:
        lo_cap, hi_cap = float(theo_lcl), float(theo_ucl)
        cap_label = "Capabilidade (teórico, σ within)"

    cap = capability_indices(s, lo_cap, hi_cap)

    # Cabeçalho do bloco de capabilidade (só mostra se algum índice foi marcado)
    if any(k in parametros_legenda for k in ["Cp","Cpk","Pp","Ppk","Cpu","Cpl","Ppu","Ppl","Zbench"]):
        handles.append(Patch(facecolor="none", edgecolor="none", label=cap_label))

    def add_cap(metric_key: str, label_pt: str):
        if label_pt not in parametros_legenda:
            return
        v = cap.get(metric_key, np.nan)
        txt = fmt_val(v, sig=3)
        handles.append(Patch(facecolor="none", edgecolor="none", label=f"{label_pt}: {txt}"))

    add_cap("Cp",    "Cp")
    add_cap("Cpk",   "Cpk")
    add_cap("Pp",    "Pp")
    add_cap("Ppk",   "Ppk")
    add_cap("Cpu",   "Cpu")
    add_cap("Cpl",   "Cpl")
    add_cap("Ppu",   "Ppu")
    add_cap("Ppl",   "Ppl")
    add_cap("Zbench","Zbench")

    # espaço para a legenda à direita
    plt.subplots_adjust(right=0.78)
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=True)
    plt.tight_layout()

    # Métricas retornadas
    metrics = {
        "Título": title,
        "N": int(s.size),
        "Mean": media,
        "Std": desvio,
        "CV_%": cv_pct,
        "Min": vmin, "Max": vmax,
        "Lo": lo, "Hi": hi,
        "Theo_LCL": float(theo_lcl) if np.isfinite(theo_lcl) else np.nan,
        "Theo_UCL": float(theo_ucl) if np.isfinite(theo_ucl) else np.nan,
        "Cap_usando_teorico": bool(use_theoretical_for_capability and np.isfinite(theo_lcl) and np.isfinite(theo_ucl)),
        "%_dentro": pct_dentro, "%_abaixo": pct_abaixo, "%_acima": pct_acima,
        "%_dentro_teo": pct_dentro_teo, "%_abaixo_teo": pct_abaixo_teo, "%_acima_teo": pct_acima_teo,
        **cap,
    }
    return fig, metrics

# ========= Sidebar =========
st.sidebar.header("Configurações globais")
bins = st.sidebar.slider("Nº bins", 10, 100, 40, 5)
col_bars = st.sidebar.color_picker("Cor barras", "#4B91CC")
col_lo = st.sidebar.color_picker("Cor LO", "#C0392B")
col_hi = st.sidebar.color_picker("Cor HI", "#6C3483")
show_norm = st.sidebar.checkbox("Curva normal", True)
save_zip = st.sidebar.checkbox("Zip figuras", False)
PALETTE = {"bars": col_bars, "lo": col_lo, "hi": col_hi}

# ---- Limites teóricos (SPC 3σ within) — VISUAL ----
st.sidebar.subheader("Limites teóricos (3σ within) — visual")
use_theo = st.sidebar.checkbox("Sobrepor LCL/UCL teóricos (3σ within)", value=False)
use_theo_for_cap = st.sidebar.checkbox("Usar teóricos no cálculo da capabilidade (comparação)", value=False)
col_theo_line = st.sidebar.color_picker("Cor das linhas teóricas", "#1F618D")
col_theo_fill = st.sidebar.color_picker("Cor da faixa teórica (fundo)", "#D6EAF8")
theo_alpha = st.sidebar.slider("Opacidade da faixa teórica", 0.0, 0.6, 0.18, 0.02)

# ========================
# Seção 1 — Upload CSVs (persistente)
# ========================
st.subheader("1) Envie os **dois** arquivos CSV")

# Inicializa chaves na sessão
for k in ("lim_bytes", "lim_name", "qua_bytes", "qua_name"):
    st.session_state.setdefault(k, None)

col_up1, col_up2 = st.columns(2)
with col_up1:
    up_lim = st.file_uploader("Catálogo (Limites)", type="csv", key="uplim")
    if up_lim is not None:
        st.session_state["lim_bytes"] = up_lim.getvalue()
        st.session_state["lim_name"] = up_lim.name
with col_up2:
    up_qua = st.file_uploader("Dados (Qualidade)", type="csv", key="upqua")
    if up_qua is not None:
        st.session_state["qua_bytes"] = up_qua.getvalue()
        st.session_state["qua_name"] = up_qua.name

cols_info = st.columns([3,1])
with cols_info[0]:
    if st.session_state["lim_name"] and st.session_state["qua_name"]:
        st.success(f"Carregado: **{st.session_state['lim_name']}** e **{st.session_state['qua_name']}**")
    else:
        st.info("Envie os dois CSVs para continuar.")
with cols_info[1]:
    if st.button("Limpar arquivos"):
        for k in ("lim_bytes", "lim_name", "qua_bytes", "qua_name"):
            st.session_state[k] = None
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

have_files = st.session_state["lim_bytes"] is not None and st.session_state["qua_bytes"] is not None

if have_files:
    # Lê a partir dos bytes persistidos
    try:
        catalog_df = read_csv_from_bytes(st.session_state["lim_bytes"])
        data_df    = read_csv_from_bytes(st.session_state["qua_bytes"])
    except Exception as e:
        st.error(f"Falha ao ler os CSVs: {e}")
        st.stop()

    # Checa colunas
    if "Tag" not in catalog_df.columns:
        st.error("O CSV de Limites precisa ter a coluna 'Tag' (e opcionalmente 'Nome', 'LO', 'HI').")
        st.stop()
    # Mapeia Nome (se ausente, usa Tag)
    catalog_df["Nome"] = catalog_df.get("Nome", catalog_df["Tag"]).astype(str)
    catalog_df["Tag"] = catalog_df["Tag"].astype(str)

    tags = catalog_df["Tag"].str.strip().unique().tolist()
    data_df = coerce_numeric_columns(data_df, tags)

    # Dicionários de exibição e lookup
    nome_por_tag = {row["Tag"]: str(row["Nome"]) for _, row in catalog_df.iterrows()}
    display_por_tag = {t: f"{nome_por_tag.get(t, t)} ({t})" for t in tags}
    # Listas ordenadas por nome de exibição
    options_hist = sorted([(display_por_tag[t], t) for t in tags], key=lambda x: x[0])

    # ========================
    # Seção 2 — Filtros e seleção de variáveis
    # ========================
    st.subheader("2) Filtros e seleção de variáveis")

    # ---- 2.1 Filtro GLOBAL (somente Produção) ----
    st.markdown("#### 2.1) Filtro global — **Produção**")
    tag_producao = st.selectbox(
        "Escolha a **Tag de Produção** (filtro global):",
        options=[lbl for lbl, t in options_hist],
        index=0 if options_hist else None,
        key="tag_producao_lbl"
    )
    tag_producao_real = next((t for lbl, t in options_hist if lbl == tag_producao), None)

    colp = st.columns([1,1,2])
    op_prod = colp[0].selectbox("Operador (Produção)", [">", ">=", "<", "<=", "==", "!=", "entre"], key="op_prod")
    v1_prod = colp[1].number_input("V1 (Produção)", key="v1_prod")
    v2_prod = None
    if op_prod == "entre":
        v2_prod = colp[2].number_input("V2 (Produção)", key="v2_prod")

    # Aplica APENAS o filtro de produção, globalmente
    df_after_prod = data_df.copy()
    if tag_producao_real in df_after_prod.columns:
        if op_prod == ">":   df_after_prod = df_after_prod[df_after_prod[tag_producao_real] >  v1_prod]
        elif op_prod == ">=":df_after_prod = df_after_prod[df_after_prod[tag_producao_real] >= v1_prod]
        elif op_prod == "<": df_after_prod = df_after_prod[df_after_prod[tag_producao_real] <  v1_prod]
        elif op_prod == "<=":df_after_prod = df_after_prod[df_after_prod[tag_producao_real] <= v1_prod]
        elif op_prod == "==":df_after_prod = df_after_prod[df_after_prod[tag_producao_real] == v1_prod]
        elif op_prod == "!=":df_after_prod = df_after_prod[df_after_prod[tag_producao_real] != v1_prod]
        elif op_prod == "entre" and v2_prod is not None:
            lo_v, hi_v = sorted([v1_prod, v2_prod])
            df_after_prod = df_after_prod[(df_after_prod[tag_producao_real] >= lo_v) & (df_after_prod[tag_producao_real] <= hi_v)]

    st.markdown(f"**Amostras após filtro de Produção:** {len(df_after_prod)}")

    # ---- 2.2 Filtros POR VARIÁVEL (não globais) ----
    st.markdown("#### 2.2) Filtros por variável (aplicados somente ao respectivo histograma)")
    filtro_escolhas = st.multiselect(
        "Variáveis a filtrar (locais)",
        options=[lbl for lbl, t in options_hist if t != tag_producao_real],
        default=[]
    )
    filtro_tags = [t for lbl, t in options_hist if lbl in filtro_escolhas]

    # Guardar as condições por tag (para aplicar depois, no loop de plotagem)
    conds_por_tag = {}
    for tag in filtro_tags:
        lbl = display_por_tag[tag]
        cols = st.columns([1,1,2])
        op = cols[0].selectbox(f"Operador — {lbl}", [">", ">=", "<", "<=", "==", "!=", "entre"], key=f"op_{tag}")
        v1 = cols[1].number_input(f"V1 — {lbl}", key=f"v1_{tag}")
        v2 = None
        if op == "entre":
            v2 = cols[2].number_input(f"V2 — {lbl}", key=f"v2_{tag}")
        conds_por_tag[tag] = (op, v1, v2)

    # ---- 2.3 Seleção de variáveis para histogramas ----
    vars_hist_labels = st.multiselect(
        "Variáveis para histogramas",
        options=[lbl for lbl, t in options_hist],
        default=[lbl for lbl, t in options_hist]
    )
    vars_hist = [t for lbl, t in options_hist if lbl in vars_hist_labels]

    # ---- 2.4 Remoção de outliers por variável ----
    st.markdown("#### 2.3) Remoção de outliers (por variável, após Produção)")
    outlier_cfg = {}
    with st.expander("Configurar remoção de outliers por variável", expanded=False):
        for lbl, t in options_hist:
            if t not in [x for x in vars_hist]:  # só mostra config de quem vai plotar
                continue
            row = st.columns([1,1,1,1.2])
            ativa = row[0].checkbox(f"Remover outliers — {lbl}", value=False, key=f"ol_on_{t}")
            metodo = row[1].selectbox("Método", ["IQR", "Z-score"], key=f"ol_m_{t}")
            k = row[2].number_input("k (IQR)", min_value=0.5, max_value=5.0, value=1.5, step=0.1, key=f"ol_k_{t}")
            zthr = row[3].number_input("|z| ≤ (Z-score)", min_value=1.0, max_value=8.0, value=3.0, step=0.5, key=f"ol_z_{t}")
            outlier_cfg[t] = {"on": ativa, "method": metodo, "k": k, "z": zthr}

    # ========================
    # Seção 3 — Pré-visualização
    # ========================
    st.subheader("3) Pré-visualização")
    c1, c2 = st.columns(2)
    c1.markdown("**Limites**"); c1.dataframe(catalog_df[["Tag","Nome","LO","HI"]].head(20), use_container_width=True)
    c2.markdown("**Qualidade**"); c2.dataframe(data_df.head(10), use_container_width=True)

    # ========================
    # Seção 3.1 — Parâmetros da legenda (PT-BR)
    # ========================
    st.subheader("3.1) Parâmetros da legenda")
    parametros_legenda_possiveis = [
        "Média", "Desvio-padrão", "Coeficiente de variação",
        "Mínimo", "Máximo",
        "% dentro", "% abaixo", "% acima",
        "% dentro (teórico)", "% abaixo (teórico)", "% acima (teórico)",
        "Cp", "Cpk", "Pp", "Ppk", "Cpu", "Cpl", "Ppu", "Ppl", "Zbench"
    ]
    parametros_legenda = st.multiselect(
        "Escolha o que mostrar na legenda (tudo marcado por padrão):",
        options=parametros_legenda_possiveis,
        default=parametros_legenda_possiveis
    )

    # ========================
    # Seção 4 — Geração de histogramas
    # ========================
    if st.button("Gerar histogramas"):
        stats: List[Dict] = []; imgs: Dict[str, bytes] = {}
        with st.spinner("Gerando gráficos e métricas…"):
            for _, row in catalog_df[catalog_df["Tag"].isin(vars_hist)].iterrows():
                tag = str(row["Tag"]).strip()
                nome = str(row.get("Nome", tag)).strip()
                titulo = f"{nome} ({tag})"

                # Limites do catálogo
                lo_raw, hi_raw = row.get("LO"), row.get("HI")
                lo = None if (lo_raw is None or (isinstance(lo_raw, str) and lo_raw.strip() == "") or pd.isna(lo_raw)) else float(lo_raw)
                hi = None if (hi_raw is None or (isinstance(hi_raw, str) and hi_raw.strip() == "") or pd.isna(hi_raw)) else float(hi_raw)

                if tag not in df_after_prod.columns:
                    st.warning(f"Tag não encontrada: {tag}")
                    continue

                # Base: somente Produção já filtrada
                df_local = df_after_prod.copy()

                # Aplica filtro LOCAL (se houver) para ESTA tag
                if tag in conds_por_tag:
                    op, v1, v2 = conds_por_tag[tag]
                    if op == ">":   df_local = df_local[df_local[tag] >  v1]
                    elif op == ">=":df_local = df_local[df_local[tag] >= v1]
                    elif op == "<": df_local = df_local[df_local[tag] <  v1]
                    elif op == "<=":df_local = df_local[df_local[tag] <= v1]
                    elif op == "==":df_local = df_local[df_local[tag] == v1]
                    elif op == "!=":df_local = df_local[df_local[tag] != v1]
                    elif op == "entre" and v2 is not None:
                        lo_v, hi_v = sorted([v1, v2])
                        df_local = df_local[(df_local[tag] >= lo_v) & (df_local[tag] <= hi_v)]

                # Série para plot: numérica, sem NaN
                serie_plot = pd.to_numeric(df_local[tag], errors="coerce").dropna()

                # *** Regra adicional: FORÇAR > 0 por variável ***
                n0_before = len(serie_plot)
                serie_plot = serie_plot[serie_plot > 0]
                n0_removed = n0_before - len(serie_plot)

                # Remoção de outliers (por variável), após Produção e >0
                n_before = len(serie_plot)
                cfg = outlier_cfg.get(tag, {"on": False})
                if cfg.get("on", False) and not serie_plot.empty:
                    if cfg["method"] == "IQR":
                        serie_plot = remove_outliers_series(serie_plot, method="IQR", k=float(cfg["k"]))
                    else:
                        serie_plot = remove_outliers_series(serie_plot, method="Z-score", zthr=float(cfg["z"]))
                n_after = len(serie_plot)

                # (Opcional) avisos
                if n0_removed > 0:
                    st.caption(f"**{titulo}** — removidos {n0_removed} valores ≤ 0 após filtro de Produção.")
                if cfg.get("on", False):
                    st.caption(f"**{titulo}** — outliers removidos: {n_before - n_after} (restaram {n_after}).")

                # Se a série ficou vazia, avisa e segue
                if serie_plot.empty:
                    st.warning(f"Sem dados válidos (>0) para **{titulo}** após filtros/outliers.")
                    continue

                fig, met = gen_hist(
                    serie_plot, titulo,
                    lo, hi, bins, PALETTE, show_norm,
                    use_theoretical=use_theo,
                    theo_line_color=col_theo_line,
                    theo_fill_color=col_theo_fill,
                    theo_fill_alpha=theo_alpha,
                    use_theoretical_for_capability=use_theo_for_cap,
                    parametros_legenda=parametros_legenda,
                )
                st.pyplot(fig)

                # Salvar imagem com "Nome_Tag.png"
                if save_zip:
                    fname = slug(f"{nome}_{tag}") + ".png"
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                    imgs[fname] = buf.getvalue()

                plt.close(fig)
                stats.append(met)

        if stats:
            df_stats = pd.DataFrame(stats)
            # Ordenação amigável (inclui Min/Max)
            preferred = [
                "Título", "N", "Mean", "Std", "CV_%", "Min", "Max",
                "Lo", "Hi", "Theo_LCL", "Theo_UCL", "Cap_usando_teorico",
                "%_dentro", "%_abaixo", "%_acima",
                "%_dentro_teo", "%_abaixo_teo", "%_acima_teo",
                "sd_within", "Cp", "Cpk", "Pp", "Ppk",
                "Cpu", "Cpl", "Ppu", "Ppl", "Zbench"
            ]
            ordered_cols = [c for c in preferred if c in df_stats.columns] + \
                           [c for c in df_stats.columns if c not in preferred]
            df_stats = df_stats[ordered_cols]

            st.markdown("**Métricas**")
            st.dataframe(df_stats, use_container_width=True)

            # --- Downloads XLSX ---
            def df_to_xlsx_bytes(df, sheet_name):
                b = io.BytesIO()
                with pd.ExcelWriter(b, engine="openpyxl") as w:
                    df.to_excel(w, index=False, sheet_name=sheet_name)
                b.seek(0)
                return b.getvalue()

            # 1) Todas as métricas
            st.download_button(
                "Baixar métricas (XLSX)",
                df_to_xlsx_bytes(df_stats, "Metricas"),
                "metricas_histogramas.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            # 2) Capabilidade
            cap_cols = [
                "Título", "N", "Mean", "Std", "CV_%", "Min", "Max",
                "sd_within", "Lo", "Hi",
                "Theo_LCL", "Theo_UCL", "Cap_usando_teorico",
                "%_dentro", "%_abaixo", "%_acima",
                "%_dentro_teo", "%_abaixo_teo", "%_acima_teo",
                "Cp", "Cpk", "Pp", "Ppk", "Cpu", "Cpl", "Ppu", "Ppl", "Zbench"
            ]
            cap_cols = [c for c in cap_cols if c in df_stats.columns]
            df_cap = df_stats[cap_cols]

            st.download_button(
                "Baixar capabilidade (XLSX)",
                df_to_xlsx_bytes(df_cap, "Capabilidade"),
                "capabilidade.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            # 3) Figuras (se ativado)
            if save_zip and imgs:
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, content in imgs.items():
                        zf.writestr(fname, content)
                zbuf.seek(0)
                st.download_button(
                    "Baixar figuras (.zip)",
                    data=zbuf.getvalue(),
                    file_name="figuras_histogramas.zip",
                    mime="application/zip",
                    use_container_width=True
                )
else:
    st.info("Envie os dois CSVs para continuar.")

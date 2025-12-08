# app.py — SynthAI Pro Max — FINAL PROFESSIONAL & USER-CONTROLLED (2025)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import chardet
import csv
from datetime import datetime
import warnings
import io
import zipfile

warnings.filterwarnings("ignore")
plt.style.use('default')
sns.set_palette("husl")

# ---------------------- SDV Imports ----------------------
try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality
except Exception:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    evaluate_quality = None

# ---------------------- PROFESSIONAL PREPROCESSING ----------------------
def prepare_data_for_synthesis(raw_df):
    df = raw_df.copy()
    preserve_cols = []
    drop_from_synthesis = []

    for col in df.columns:
        col_low = col.lower()

        if any(k in col_low for k in ['id', 'date', 'time', 'timestamp', 'name', 'email', 'phone', 'address', 'description', 'comment', 'sku', 'code', 'url']):
            preserve_cols.append(col)
            continue

        if df[col].dtype == 'object':
            if df[col].nunique(dropna=True) > 50:
                preserve_cols.append(col)
                continue

        if df[col].nunique(dropna=True) <= 1 or df[col].isna().mean() > 0.99:
            drop_from_synthesis.append(col)

    synth_cols = [c for c in df.columns if c not in preserve_cols and c not in drop_from_synthesis]
    clean_df = df[synth_cols].copy()
    clean_df = clean_df.dropna(how='any').reset_index(drop=True)

    for col in clean_df.select_dtypes(include=['object']).columns:
        counts = clean_df[col].value_counts()
        rare = counts[counts < 3].index
        if len(rare) > 0:
            clean_df[col] = clean_df[col].replace(rare, 'OTHER')

    return clean_df, preserve_cols

def coerce_numeric_safely(df):
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors='coerce')
        if numeric.isna().mean() < 0.4:
            median = numeric.median()
            df[col] = numeric.fillna(median if not pd.isna(median) else 0.0).astype(float)
        else:
            df[col] = df[col].fillna('missing').astype(str)
    return df

# ---------------------- MODEL GENERATION ----------------------
def generate_synthetic(clean_df, n_rows, model_choice, user_epochs=300, user_batch=128):
    messages = [f"Clean rows: {len(clean_df)} | Synthesis columns: {len(clean_df.columns)}"]

    if model_choice == "GaussianCopula (Fast & Reliable)":
        messages.append("Using GaussianCopula as requested")
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(clean_df)
        model = GaussianCopulaSynthesizer(meta)
        model.fit(clean_df)
        synth = model.sample(n_rows)
        return synth, "GaussianCopula", messages

    # CTGAN path
    if len(clean_df) < 30:
        messages.append("Dataset too small for CTGAN → using GaussianCopula")
        meta = SingleTableMetadata(); meta.detect_from_dataframe(clean_df)
        model = GaussianCopulaSynthesizer(meta); model.fit(clean_df)
        return model.sample(n_rows), "GaussianCopula", messages

    max_cats = max([clean_df[col].nunique() for col in clean_df.select_dtypes('object').columns], default=0)
    batch_size = user_batch
    if max_cats > batch_size:
        batch_size = max(64, ((max_cats // 32) + 1) * 32)
        messages.append(f"High-cardinality → batch_size increased to {batch_size}")

    batch_size = min(batch_size, len(clean_df))
    batch_size = (batch_size // 32) * 32 or 32

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(clean_df)
    for col in clean_df.select_dtypes('object').columns:
        meta.update_column(col, sdtype='categorical')

    try:
        ctgan = CTGANSynthesizer(
            meta,
            epochs=min(user_epochs, 500),
            batch_size=batch_size,
            enforce_min_max_values=True,
            enforce_rounding=False,
            verbose=False,
            pac=1
        )
        ctgan.fit(clean_df)
        synth = ctgan.sample(n_rows)
        messages.append("CTGAN training succeeded!")
        return synth, "CTGAN", messages
    except Exception as e:
        messages.append(f"CTGAN failed → fallback to GaussianCopula ({repr(e)})")
        meta = SingleTableMetadata(); meta.detect_from_dataframe(clean_df)
        model = GaussianCopulaSynthesizer(meta); model.fit(clean_df)
        return model.sample(n_rows), "GaussianCopula (fallback)", messages

# ---------------------- UI ----------------------
st.set_page_config(page_title="SynthAI Pro Max", layout="wide", page_icon="gem")
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    * {font-family: 'Inter', sans-serif;}
    .main {background: linear-gradient(135deg,#0f172a 0%,#1e293b 100%); color: #e2e8f0;}
    .header {background: linear-gradient(135deg,#7c3aed 0%,#ec4899 100%); padding: 3rem; border-radius: 24px; text-align: center; margin-bottom: 2rem; box-shadow: 0 20px 50px rgba(124,58,237,0.5);}
    .header h1 {font-size: 4rem; margin:0; background: linear-gradient(90deg,#fff,#ddd6fe); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
    .card {background: rgba(30,41,59,0.95); border-radius: 16px; padding: 2rem; margin: 1.5rem 0; border: 1px solid rgba(139,92,246,0.3);}
    .stButton>button {background: linear-gradient(90deg,#7c3aed,#ec4899); color: white; font-weight:700; border-radius: 12px; padding: 1rem 2rem;}
</style>""", unsafe_allow_html=True)

st.markdown("<div class='header'><h1>SynthAI Pro Max</h1><p>Never Fails • Preserves Real IDs & Text • Full User Control</p></div>", unsafe_allow_html=True)

for k in ['original','clean','preserve','synthetic','hybrid','model_used','quality','messages']:
    if k not in st.session_state:
        st.session_state[k] = None

tab1, tab2, tab3, tab4 = st.tabs(["1. Upload & Summary", "2. Configure", "3. Generate", "4. Validation"])

# ---------------------- TAB 1 ----------------------
with tab1:
    st.markdown("<div class='card'><h2>Upload CSV or Excel</h2></div>", unsafe_allow_html=True)
    file = st.file_uploader("Choose file", type=["csv","xlsx","xls"])
    if file:
        try:
            if file.name.endswith(('.xlsx','.xls')):
                df_raw = pd.read_excel(file)
            else:
                enc = chardet.detect(file.read(100000))['encoding'] or 'utf-8'
                file.seek(0)
                sample = file.read(1024*1024).decode(enc, errors='ignore')
                file.seek(0)
                delim = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|']).delimiter
                df_raw = pd.read_csv(file, encoding=enc, sep=delim, on_bad_lines='skip')

            df_raw = df_raw.dropna(how='all').reset_index(drop=True)
            df_clean, preserve_cols = prepare_data_for_synthesis(df_raw)
            df_clean = coerce_numeric_safely(df_clean)

            for col in df_clean.columns:
                if str(df_clean[col].dtype) == 'category':
                    df_clean[col] = df_clean[col].astype(str)

            st.session_state.original = df_raw.copy()
            st.session_state.clean = df_clean.copy()
            st.session_state.preserve = df_raw[preserve_cols].copy() if preserve_cols else pd.DataFrame()

            st.success("Data ready!")
            c1,c2,c3 = st.columns(3)
            c1.metric("Original Rows", len(df_raw))
            c2.metric("Rows for Synthesis", len(df_clean))
            c3.metric("Preserved Columns", len(preserve_cols))

            with st.expander("Preview Clean Data", expanded=True):
                st.dataframe(df_clean.head(200), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------- TAB 2 — NOW WITH MODEL SELECTION ----------------------
with tab2:
    st.markdown("<div class='card'><h2>Configure Synthesis</h2></div>", unsafe_allow_html=True)
    if st.session_state.clean is None:
        st.info("Upload data first"); st.stop()

    # USER CAN NOW CHOOSE THE MODEL
    st.session_state.model_choice = st.radio(
        "Select Synthesis Model",
        ["CTGAN (Best Quality)", "GaussianCopula (Fast & Reliable)"],
        index=0,
        help="CTGAN = highest fidelity. GaussianCopula = faster & always works."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.n_rows = st.slider("Synthetic rows", 100, 500000, 10000, step=1000)
        st.session_state.threshold = st.slider("Quality threshold", 0.6, 0.99, 0.90, 0.01)
    with c2:
        st.session_state.epochs = st.slider("CTGAN Epochs (only used if CTGAN selected)", 100, 500, 300)
        st.session_state.batch_size = st.select_slider("Batch Size (CTGAN only)", [32,64,128,256,512], 128)

# ---------------------- TAB 3 ----------------------
with tab3:
    st.markdown("<div class='card'><h2>Generate Hybrid Data</h2></div>", unsafe_allow_html=True)
    if st.session_state.clean is None:
        st.info("Upload first"); st.stop()

    if st.button("GENERATE NOW", type="primary"):
        with st.spinner("Training model..."):
            synth_core, model_used, messages = generate_synthetic(
                st.session_state.clean,
                st.session_state.n_rows,
                st.session_state.model_choice,
                st.session_state.epochs,
                st.session_state.batch_size
            )

            if not st.session_state.preserve.empty:
                real_sample = st.session_state.preserve.sample(n=len(synth_core), replace=True, random_state=42).reset_index(drop=True)
                final_hybrid = pd.concat([synth_core.reset_index(drop=True), real_sample], axis=1)
            else:
                final_hybrid = synth_core.copy()

            score = None
            if evaluate_quality and len(st.session_state.clean) > 50:
                try:
                    meta = SingleTableMetadata(); meta.detect_from_dataframe(st.session_state.clean)
                    score = evaluate_quality(st.session_state.clean, synth_core, meta).get_score()
                except: pass

            st.session_state.synthetic = final_hybrid
            st.session_state.model_used = model_used
            st.session_state.quality = score
            st.session_state.messages = messages

            st.success(f"{model_used} completed!")
            if score: st.metric("Quality Score", f"{score:.1%}")
            if score and score >= st.session_state.threshold: st.balloons()

        with st.expander("Downloads", expanded=True):
            st.download_button("Synthetic Core CSV", synth_core.to_csv(index=False), "synthetic_core.csv")
            excel_io = io.BytesIO()
            with pd.ExcelWriter(excel_io, engine='xlsxwriter') as writer:
                final_hybrid.to_excel(writer, index=False)
            st.download_button("Full Hybrid Excel", excel_io.getvalue(), "hybrid_dataset.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------- TAB 4 — FULL VALIDATION ----------------------
with tab4:
    st.markdown("<div class='card'><h2>Validation & Plots</h2></div>", unsafe_allow_html=True)
    if st.session_state.synthetic is None:
        st.info("Generate data first"); st.stop()

    real = st.session_state.clean
    synth = st.session_state.synthetic

    if st.session_state.quality:
        st.metric("Quality Score", f"{st.session_state.quality:.1%}")

    with st.expander("Preview First 10 Rows: Real vs Synthetic", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real Data (Cleaned)**")
            st.dataframe(real.head(10), use_container_width=True)
        with c2:
            st.markdown("**Synthetic Hybrid Data**")
            st.dataframe(synth.head(10), use_container_width=True)

    num_cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c])]
    if num_cols:
        st.markdown("### Statistical Comparison (Mean, Std, KS Test)")
        stats = []
        for col in num_cols[:15]:
            real_vals = real[col].dropna()
            synth_vals = synth[col].dropna()
            ks_p = "N/A"
            if len(real_vals) > 1 and len(synth_vals) > 1:
                ks_p = round(ks_2samp(real_vals, synth_vals).pvalue, 4)
            stats.append({
                "Column": col,
                "Real Mean": round(real_vals.mean(), 3) if len(real_vals)>0 else "N/A",
                "Synth Mean": round(synth_vals.mean(), 3) if len(synth_vals)>0 else "N/A",
                "Real Std": round(real_vals.std(), 3) if len(real_vals)>0 else "N/A",
                "Synth Std": round(synth_vals.std(), 3) if len(synth_vals)>0 else "N/A",
                "KS p-value": ks_p
            })
        st.dataframe(pd.DataFrame(stats), use_container_width=True)

    cat_cols = [c for c in real.columns if real[c].dtype == 'object']
    c1, c2 = st.columns(2)
    with c1:
        plot_num = st.multiselect("Numeric", num_cols, default=num_cols[:6])
    with c2:
        plot_cat = st.multiselect("Categorical", cat_cols, default=cat_cols[:6])

    for col in (plot_num or num_cols[:6]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        sns.histplot(real[col], kde=True, ax=ax1, color="#8b5cf6")
        sns.histplot(synth[col], kde=True, ax=ax2, color="#ec4899")
        ax1.set_title(f"{col} – Real"); ax2.set_title(f"{col} – Synthetic")
        st.pyplot(fig); plt.close()

    for col in (plot_cat or cat_cols[:6]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        real[col].value_counts().head(10).plot.bar(ax=ax1, color="#8b5cf6")
        synth[col].value_counts().head(10).plot.bar(ax=ax2, color="#ec4899")
        ax1.set_title(f"{col} – Real"); ax2.set_title(f"{col} – Synthetic")
        st.pyplot(fig); plt.close()

    if len(num_cols) > 1:
        st.markdown("### Correlation Heatmaps")
        c1,c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            sns.heatmap(real[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Real")
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots()
            sns.heatmap(synth[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Synthetic")
            st.pyplot(fig); plt.close()

    with st.expander("Training Log"):
        for m in st.session_state.messages:
            st.write("• " + m)

st.markdown("<center style='margin-top:5rem; color:#94a3b8'>© 2025 SynthAI Pro Max — Professional • Unbreakable • Full User Control</center>", unsafe_allow_html=True)

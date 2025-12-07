# app.py — SynthAI Pro Max — Upgraded with Smart CTGAN Validation, Fallbacks & Downloads
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
import tempfile
import zipfile
import json
import os

warnings.filterwarnings("ignore")
plt.style.use('default')
sns.set_palette("husl")

# ---------------------- Robust SDV imports ----------------------
CTGAN_AVAILABLE = True
try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
except Exception:
    try:
        from sdv.tabular import CTGANSynthesizer, GaussianCopulaSynthesizer
    except Exception:
        CTGAN_AVAILABLE = False
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
        except Exception:
            GaussianCopulaSynthesizer = None  # will raise at runtime if used

from sdv.metadata import SingleTableMetadata
try:
    from sdv.evaluation.single_table import evaluate_quality
except Exception:
    evaluate_quality = None

# ---------------------- Utility helpers ----------------------
def smart_epoch_for_size(n_rows, user_epochs):
    """Return an advisable epochs value depending on dataset size and user setting"""
    if n_rows < 50:
        return 0  # signal: do not run CTGAN
    if n_rows < 100:
        return min(user_epochs, 80)
    if n_rows < 500:
        return min(user_epochs, 200)
    if n_rows < 2000:
        return min(user_epochs, 300)
    return min(user_epochs, 500)

def smart_batch_for_size(n_rows, user_batch):
    """Return a batch size that is multiple of 32 and fits dataset"""
    if n_rows <= 100:
        base = 32
    elif n_rows <= 500:
        base = 64
    elif n_rows <= 2000:
        base = 128
    else:
        base = 256
    # respect user's upper bound
    batch = min(user_batch, base)
    # ensure divisibility by 32
    batch = max(32, batch - (batch % 32))
    if batch == 0:
        batch = 32
    # final clip
    if batch > n_rows:
        batch = 32 if n_rows >= 32 else n_rows
    return int(batch)

def dataset_summary(df):
    """Return summary dict: total variables, missing %, per-col missing, dtypes"""
    total_vars = df.shape[1]
    total_rows = df.shape[0]
    missing_perc = df.isna().mean().mean()  # overall fraction
    per_col_missing = (df.isna().sum() / max(1, total_rows)).round(4).to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    return {
        "total_rows": total_rows,
        "total_vars": total_vars,
        "overall_missing_fraction": round(missing_perc,4),
        "per_col_missing": per_col_missing,
        "dtypes": dtypes
    }

def prepare_clean_df(raw_df):
    """
    Perform cleaning & column dropping logic:
    - drop date/time/id-like columns
    - drop object cols if high-cardinality or long strings
    - drop constant or all-missing columns
    - drop rows with any missing (keeps same prior behavior)
    Return: (clean_df, dropped_columns_list)
    """
    df = raw_df.copy()
    drop_cols = []
    for col in df.columns:
        low = col.lower()
        if any(x in low for x in ['date','time','id']):
            drop_cols.append(col)
            continue
        # dtype/object checks
        try:
            if df[col].dtype == 'object':
                if df[col].nunique(dropna=True) > 50:
                    drop_cols.append(col); continue
                if df[col].astype(str).str.len().mean() > 40:
                    drop_cols.append(col); continue
        except Exception:
            drop_cols.append(col); continue
        # drop low-variance or all-missing
        try:
            if df[col].nunique(dropna=True) <= 1 or df[col].isna().all():
                drop_cols.append(col); continue
        except Exception:
            pass
    clean_df = df.drop(columns=drop_cols, errors='ignore').reset_index(drop=True)
    # drop rows with any missing (same as your original behavior)
    clean_df = clean_df.dropna(how='any').reset_index(drop=True)
    return clean_df, drop_cols

def coerce_types_for_sdv(clean_df):
    """Force numeric columns to float and object columns to string and fillna"""
    df = clean_df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("missing").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # fill remaining NA with median if possible
            if df[col].isna().any():
                med = df[col].median()
                if pd.isna(med):
                    med = 0.0
                df[col] = df[col].fillna(med)
            # cast ints to float
            if pd.api.types.is_integer_dtype(df[col].dtype):
                df[col] = df[col].astype(float)
    return df

def detect_high_cardinality(df, threshold=100):
    warnings_list = []
    for col in df.columns:
        try:
            if df[col].dtype == 'object' and df[col].nunique(dropna=True) > threshold:
                warnings_list.append(f"High-cardinality column: {col} ({df[col].nunique()} unique)")
        except Exception:
            warnings_list.append(f"Could not evaluate cardinality for: {col}")
    return warnings_list

def generate_zip_package(files_dict):
    """
    files_dict: dict(filename -> bytes)
    returns: bytes of zip
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in files_dict.items():
            z.writestr(name, content)
    buf.seek(0)
    return buf.read()

# ---------------------- CTGAN training wrapper with auto-fallback ----------------------
def try_fit_ctgan(clean_df, n_rows, epochs, batch_size, generator_decay=1e-5, discriminator_decay=1e-5):
    """
    Attempt CTGAN training with smart scaling and safety checks.
    Returns dict with keys: model ('CTGAN'/'GaussianCopula'), synthesizer, synthetic_core, messages
    May fallback to GaussianCopulaSynthesizer automatically.
    """
    messages = []
    result = {"model": None, "synthesizer": None, "synthetic_core": None, "messages": messages}

    if not CTGAN_AVAILABLE:
        messages.append("CTGAN not available in SDV installation; using GaussianCopula.")
        result["model"] = "GaussianCopula"
        # try Gaussian
        meta = SingleTableMetadata(); meta.detect_from_dataframe(clean_df)
        gc = GaussianCopulaSynthesizer(meta)
        gc.fit(clean_df)
        result["synthesizer"] = gc
        result["synthetic_core"] = gc.sample(n_rows)
        return result

    n = len(clean_df)
    advised_epochs = smart_epoch_for_size(n, epochs)
    if advised_epochs == 0:
        messages.append(f"Dataset too small for CTGAN ({n} rows). Switching to GaussianCopula.")
        result["model"] = "GaussianCopula"
        meta = SingleTableMetadata(); meta.detect_from_dataframe(clean_df)
        gc = GaussianCopulaSynthesizer(meta)
        gc.fit(clean_df)
        result["synthesizer"] = gc
        result["synthetic_core"] = gc.sample(n_rows)
        return result

    advised_batch = smart_batch_for_size(n, batch_size)
    messages.append(f"CTGAN advised epochs: {advised_epochs}, batch_size: {advised_batch}")

    # extra validations
    hc_warnings = detect_high_cardinality(clean_df, threshold=200)
    if hc_warnings:
        messages.append("High-cardinality warnings: " + " | ".join(hc_warnings))
    # Coerce types again for safety
    safe_df = coerce_types_for_sdv(clean_df)

    # Create metadata and force categorical sdtypes for objects if possible
    meta = SingleTableMetadata(); meta.detect_from_dataframe(safe_df)
    for col in safe_df.columns:
        if safe_df[col].dtype == 'object':
            try:
                meta.update_column(col, sdtype='categorical')
            except Exception:
                pass

    # instantiate CTGAN
    synthesizer = CTGANSynthesizer(
        meta,
        epochs=advised_epochs,
        batch_size=advised_batch,
        generator_decay=generator_decay,
        discriminator_decay=discriminator_decay,
        enforce_min_max_values=True,
        enforce_rounding=False,
        verbose=False
    )

    try:
        synthesizer.fit(safe_df)
        synthetic_core = synthesizer.sample(n_rows)
        result.update({"model": "CTGAN", "synthesizer": synthesizer, "synthetic_core": synthetic_core})
        messages.append("CTGAN training succeeded.")
        return result
    except Exception as e:
        # log the exception and fallback
        messages.append(f"CTGAN training failed with: {repr(e)}")
        messages.append("Falling back to GaussianCopula to ensure the app returns results.")
        meta2 = SingleTableMetadata(); meta2.detect_from_dataframe(safe_df)
        gc = GaussianCopulaSynthesizer(meta2)
        gc.fit(safe_df)
        result.update({"model": "GaussianCopula (fallback)", "synthesizer": gc, "synthetic_core": gc.sample(n_rows)})
        return result

# ---------------------- UI / Layout ----------------------
st.set_page_config(page_title="SynthAI Pro Max", layout="wide", page_icon="💎")
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap');
    * {font-family: 'Inter', sans-serif;}
    .main {background: linear-gradient(135deg,#0f172a 0%,#1e293b 100%); color: #e2e8f0;}
    .header {background: linear-gradient(135deg,#7c3aed 0%,#ec4899 100%); padding: 4rem; border-radius: 32px; text-align: center; margin-bottom: 2.5rem; box-shadow: 0 30px 70px rgba(124,58,237,0.6);}
    .header h1 {font-size: 4rem; margin:0; background: linear-gradient(90deg,#fff,#ddd6fe); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:900;}
    .card {background: rgba(30,41,59,0.95); backdrop-filter: blur(8px); border-radius: 16px; padding: 1.6rem; margin: 1.2rem 0; border: 1px solid rgba(139,92,246,0.25); box-shadow: 0 12px 40px rgba(0,0,0,0.4);}
    .stButton>button {background: linear-gradient(90deg,#7c3aed,#ec4899); color: white; font-weight:700; border-radius: 12px; padding: 0.8rem 1.6rem; font-size:1rem;}
</style>""", unsafe_allow_html=True)

st.markdown("<div class='header'><h1>SynthAI Pro Max</h1><p>You Choose: CTGAN (Best Quality) or GaussianCopula (Fast & Reliable)</p></div>", unsafe_allow_html=True)

# initialize session state
for key in ['original','cleaned','dropped_cols','synthetic','n_rows','threshold','epochs','batch_size','quality_score','model','last_messages']:
    if key not in st.session_state:
        st.session_state[key] = None

tab1, tab2, tab3, tab4 = st.tabs(["Upload & Summary", "Configure", "Generate", "Validation"])

# ---------------------- TAB 1: UPLOAD & SUMMARY ----------------------
with tab1:
    st.markdown("<div class='card'><h2>1) Upload → Read → Initial Clean → Summary</h2></div>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"])
    if file:
        try:
            if file.name.lower().endswith(('.xlsx','.xls')):
                df_raw = pd.read_excel(file)
            else:
                raw = file.read()
                enc = chardet.detect(raw[:100000])['encoding'] or 'utf-8'
                file.seek(0)
                sample = file.read(1024*1024).decode(enc, errors='ignore')
                file.seek(0)
                delim = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|']).delimiter
                df_raw = pd.read_csv(file, encoding=enc, sep=delim, on_bad_lines='skip')

            df_raw = df_raw.dropna(how='all').reset_index(drop=True)
            df_raw = df_raw.loc[~(df_raw.astype(str).apply(lambda x: x.str.contains('total|sum|grand|average', case=False, na=False)).any(axis=1))]

            # prepare clean set & dropped columns (step 4 + 5)
            df_clean, drop_cols = prepare_clean_df(df_raw)
            st.session_state.original = df_raw.copy()
            st.session_state.cleaned = df_clean.copy()
            st.session_state.dropped_cols = df_raw[drop_cols].copy() if drop_cols else pd.DataFrame()
            st.session_state.last_messages = []

            # summary
            summary = dataset_summary(df_raw)
            st.success("Data uploaded & preliminarily cleaned.")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Original Rows", summary["total_rows"])
            c2.metric("Final Rows (after drop/missing removal)", df_clean.shape[0])
            c3.metric("Columns Dropped", len(drop_cols))
            c4.metric("Overall Missing %", f"{summary['overall_missing_fraction']*100:.2f}%")

            # Show dropped columns list
            if drop_cols:
                with st.expander(f"Dropped variables ({len(drop_cols)}) — click to view", expanded=False):
                    st.write(drop_cols)

            # Data preview and detailed summary
            with st.expander("Preview Clean Data & Detailed Summary", expanded=True):
                st.write("**Clean Data (first 200 rows):**")
                st.dataframe(df_clean.head(200), use_container_width=True)
                st.write("**Per-column missing fraction & types:**")
                df_types = pd.DataFrame({
                    "dtype": pd.Series(summary["dtypes"]),
                    "missing_fraction": pd.Series(summary["per_col_missing"])
                })
                st.dataframe(df_types.T if df_types.shape[0] < 8 else df_types, use_container_width=True)
        except Exception as e:
            st.error(f"Upload/parse error: {e}")

# ---------------------- TAB 2: CONFIGURE ----------------------
with tab2:
    st.markdown("<div class='card'><h2>2) Choose Model & Settings</h2></div>", unsafe_allow_html=True)
    if st.session_state.cleaned is None:
        st.info("Upload dataset on the 'Upload & Summary' tab first.")
        st.stop()

    st.session_state.model = st.radio(
        "Select Synthesis Model",
        options=["CTGAN (Neural Network – Best Quality)", "GaussianCopula (Fast & Statistical)"],
        index=0
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.n_rows = st.slider("Synthetic Rows to Generate", 100, 500000, 5000, step=100)
        st.session_state.threshold = st.slider("Quality Threshold (0.0 - 1.0)", 0.6, 0.99, 0.90, 0.01)
    with col2:
        st.session_state.epochs = st.slider("Epochs (CTGAN only)", 50, 500, 300)
        st.session_state.batch_size = st.select_slider("Batch Size (CTGAN only)", options=[32,64,128,256,512], value=128)
    st.info("Smart scaling is automatic: CTGAN epochs & batch size will adapt to dataset size. If dataset is too small, GaussianCopula will be used automatically.")

# ---------------------- TAB 3: GENERATE ----------------------
with tab3:
    st.markdown("<div class='card'><h2>3) Generate Synthetic / Hybrid Data</h2></div>", unsafe_allow_html=True)
    if st.session_state.cleaned is None:
        st.info("Upload dataset on first tab.")
        st.stop()

    if st.button("GENERATE NOW", type="primary"):
        df = st.session_state.cleaned.copy()
        messages_box = st.empty()
        with st.spinner("Preparing model..."):
            # show basic diagnostics
            total_vars = df.shape[1]
            missing_frac = df.isna().mean().mean()
            messages = [f"Clean rows: {len(df)} | Vars: {total_vars} | Missing fraction (cleaned dataset): {missing_frac:.3f}"]
            st.session_state.last_messages = messages

            # Attempt CTGAN or chosen model
            if "CTGAN" in st.session_state.model:
                res = try_fit_ctgan(df, st.session_state.n_rows, st.session_state.epochs, st.session_state.batch_size)
            else:
                # User chose GaussianCopula directly
                meta = SingleTableMetadata(); meta.detect_from_dataframe(df)
                gc = GaussianCopulaSynthesizer(meta)
                gc.fit(df)
                res = {"model": "GaussianCopula", "synthesizer": gc, "synthetic_core": gc.sample(st.session_state.n_rows), "messages": ["GaussianCopula fitted as requested."]}

            # show messages
            for m in res.get("messages", []):
                st.session_state.last_messages.append(m)
            messages_box.info("\n".join(st.session_state.last_messages))

            synthetic_core = res["synthetic_core"]
            model_used = res["model"]

            # combine with dropped columns (IDs/dates etc.)
            if not st.session_state.dropped_cols.empty:
                # sample dropped columns to match size
                real_sample = st.session_state.dropped_cols.sample(n=len(synthetic_core), replace=True, random_state=42).reset_index(drop=True)
                final = pd.concat([synthetic_core.reset_index(drop=True), real_sample], axis=1)
            else:
                final = synthetic_core.copy()

            # Quality evaluation
            score = None
            try:
                if evaluate_quality is not None:
                    meta_eval = SingleTableMetadata(); meta_eval.detect_from_dataframe(df)
                    score = evaluate_quality(df, synthetic_core, meta_eval).get_score()
                else:
                    score = None
            except Exception as e:
                st.warning(f"Quality evaluation failed: {e}")
                score = None

            st.session_state.synthetic = final
            st.session_state.quality_score = score
            st.success(f"{model_used} completed. Quality: {score:.1%}" if score is not None else f"{model_used} completed.")
            if score is not None and score >= st.session_state.threshold:
                st.balloons()

            # Save messages
            st.session_state.last_messages.append(f"Model used: {model_used}")
            if score is not None:
                st.session_state.last_messages.append(f"Quality score: {score:.4f}")

            # Provide downloads (synthetic-only, hybrid, full package)
            with st.expander("Download Options", expanded=True):
                # synthetic-only
                synth_csv = synthetic_core.to_csv(index=False).encode('utf-8')
                st.download_button("Download synthetic-only CSV", data=synth_csv,
                                   file_name=f"synthetic_core_{model_used}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")
                # hybrid
                hybrid_csv = final.to_csv(index=False).encode('utf-8')
                st.download_button("Download hybrid CSV (synthetic + real dropped cols)", data=hybrid_csv,
                                   file_name=f"hybrid_{model_used}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")
                # full package zip
                files = {
                    f"clean_data_{datetime.now().strftime('%Y%m%d')}.csv": st.session_state.cleaned.to_csv(index=False),
                    f"synthetic_core_{model_used}.csv": synthetic_core.to_csv(index=False),
                    f"hybrid_{model_used}.csv": final.to_csv(index=False),
                    "dropped_columns.json": json.dumps({"dropped": list(st.session_state.dropped_cols.columns) if not st.session_state.dropped_cols.empty else []}, indent=2),
                    "dataset_summary.json": json.dumps(dataset_summary(st.session_state.original), indent=2),
                    "training_messages.txt": "\n".join(st.session_state.last_messages)
                }
                # ensure bytes
                files_bytes = {k: (v if isinstance(v, (bytes, bytearray)) else v.encode('utf-8')) for k,v in files.items()}
                zip_bytes = generate_zip_package(files_bytes)
                st.download_button("Download FULL package (ZIP)", data=zip_bytes,
                                   file_name=f"synth_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                   mime="application/zip")

# ---------------------- TAB 4: VALIDATION ----------------------
with tab4:
    st.markdown("<div class='card'><h2>4) Validation & Graphical Representation</h2></div>", unsafe_allow_html=True)
    if st.session_state.synthetic is None:
        st.info("Generate synthetic data first on the 'Generate' tab.")
        st.stop()

    real = st.session_state.original
    synth = st.session_state.synthetic

    col1, col2 = st.columns(2)
    qscore = st.session_state.quality_score
    col1.metric("Final Quality Score", f"{qscore:.1%}" if qscore is not None else "N/A")
    col2.download_button(
        label="Download FULL HYBRID DATASET",
        data=synth.to_csv(index=False).encode('utf-8'),
        file_name=f"Synthetic_Hybrid_Dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
        help="Contains: synthesized columns + real IDs/dates preserved"
    )

    # Numeric comparisons (means, KS)
    num_cols = [c for c in real.columns if c in synth.columns and pd.api.types.is_numeric_dtype(real[c])]
    if num_cols:
        stats = []
        for col in num_cols:
            try:
                ks = ks_2samp(real[col].dropna(), synth[col].dropna())
                stats.append({"Column":col,"Real Mean":round(real[col].mean(),3),
                              "Synth Mean":round(synth[col].mean(),3),"KS p-value":round(ks.pvalue,4)})
            except Exception:
                stats.append({"Column":col,"Real Mean":round(real[col].mean(),3),
                              "Synth Mean":round(synth[col].mean(),3),"KS p-value":"err"})
        st.markdown("### Numeric column comparison (mean & KS p-value)")
        st.dataframe(pd.DataFrame(stats), use_container_width=True)

        if len(num_cols) > 1:
            st.markdown("### Correlation Heatmaps")
            c1,c2 = st.columns(2)
            with c1:
                fig,ax=plt.subplots()
                sns.heatmap(real[num_cols].corr(),annot=True,cmap="coolwarm",fmt=".2f",ax=ax)
                ax.set_title("Real")
                st.pyplot(fig); plt.close()
            with c2:
                fig,ax=plt.subplots()
                sns.heatmap(synth[num_cols].corr(),annot=True,cmap="coolwarm",fmt=".2f",ax=ax)
                ax.set_title("Synthetic")
                st.pyplot(fig); plt.close()

        st.markdown("### Distributions (first 6 numeric columns)")
        for col in num_cols[:6]:
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
            sns.histplot(real[col],kde=True,ax=ax1)
            sns.histplot(synth[col],kde=True,ax=ax2)
            ax1.set_title(f"{col} – Real")
            ax2.set_title(f"{col} – Synthetic")
            st.pyplot(fig); plt.close()

    # Categorical graphs
    cat_cols = [c for c in real.columns if c in synth.columns and real[c].dtype=='object'][:6]
    if cat_cols:
        st.markdown("### Categorical comparisons (top categories)")
        for col in cat_cols:
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
            real[col].value_counts().head(10).plot.bar(ax=ax1)
            synth[col].value_counts().head(10).plot.bar(ax=ax2)
            ax1.set_title(f"{col} – Real")
            ax2.set_title(f"{col} – Synthetic")
            st.pyplot(fig); plt.close()

    # Show dropped variables & dataset summary
    with st.expander("Dropped Variables & Dataset Summary", expanded=False):
        st.write("Dropped variables (IDs, dates, free text, etc.):")
        st.write(list(st.session_state.dropped_cols.columns) if not st.session_state.dropped_cols.empty else [])
        st.write("Original dataset summary:")
        st.json(dataset_summary(st.session_state.original))

    # Show training messages & guidance
    with st.expander("Training Messages & Guidance", expanded=True):
        msgs = st.session_state.last_messages or []
        for m in msgs:
            st.write("-", m)
        st.markdown("""
        **Guidance:**  
        - If CTGAN fallback occurred, try increasing dataset size, reduce high-cardinality text, or accept GaussianCopula.  
        - For better CTGAN results: more continuous variables, avoid extremely sparse categorical columns, and ensure >100 rows ideally.  
        - Use the ZIP package to inspect dropped columns and training messages for debugging.
        """)

st.markdown("<center style='margin-top:4rem; color:#94a3b8'>© 2025 SynthAI Pro Max — Smart CTGAN & Hybrid Synthesis</center>", unsafe_allow_html=True)


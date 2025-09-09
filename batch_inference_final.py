"""
Final Batch Inference System - Complete Working Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import time
import gc
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import plotly.graph_objects as go
from io import BytesIO

# Import data extraction
from data_extraction import read_file as original_read_file

# ==================== CONFIGURATION ====================
CKPT_DIR = "./my_finetuned_classifier"
BATCH_SIZE = 512
MAX_LENGTH = 32

CLASS_MAPPING = {
    'commercial': 'Commercial',
    'Commercial': 'Commercial',
    'residential': 'Residential', 
    'Residential': 'Residential',
    'Community & Instituitional': 'Community & Institutional',
    'Community & Institutional': 'Community & Institutional'
}

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Batch Inference System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== SESSION STATE ====================
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'process_time' not in st.session_state:
    st.session_state.process_time = 0
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.7

# ==================== STYLING ====================
st.markdown("""
<style>
    h1 { font-size: 48px; font-weight: 600; margin-bottom: 0.5rem; }
    h3 { font-size: 21px; font-weight: 500; margin-top: 2rem; }
    .stButton > button {
        background: #0071e3;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load model once and cache it"""
    try:
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)
            torch.backends.cudnn.benchmark = True
        
        cfg = AutoConfig.from_pretrained(CKPT_DIR)
        original_id2label = cfg.id2label
        
        consolidated_classes = {}
        unique_classes = set()
        for label in original_id2label.values():
            normalized = CLASS_MAPPING.get(label, label)
            unique_classes.add(normalized)
        
        for idx, label in enumerate(sorted(unique_classes)):
            consolidated_classes[idx] = label
        
        tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if torch.cuda.is_available():
            model = model.half()
            with torch.no_grad():
                for _ in range(3):
                    dummy = tokenizer(["warmup"] * 32, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
                    dummy = {k: v.to(device) for k, v in dummy.items()}
                    _ = model(**dummy)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        return model, tokenizer, original_id2label, consolidated_classes, device
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

# ==================== PROCESSING FUNCTIONS ====================
def process_files(uploaded_files):
    """Process uploaded Excel files"""
    all_dfs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getvalue())
            temp_path = tmp.name
        
        try:
            df = original_read_file(temp_path)
            if df is not None and not df.empty:
                all_dfs.append(df)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def predict_batch(texts, model, tokenizer, original_id2label, device):
    """Run batch predictions"""
    if not texts:
        return [], []
    
    all_preds = []
    all_scores = []
    
    model.eval()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            encoding = tokenizer(batch, truncation=True, padding='longest', max_length=MAX_LENGTH, return_tensors='pt')
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(**encoding)
            else:
                outputs = model(**encoding)
            
            probs = F.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            scores = torch.max(probs, dim=-1).values.cpu().numpy()
            
            for p, s in zip(preds, scores):
                label = original_id2label.get(int(p), "Unknown")
                all_preds.append(CLASS_MAPPING.get(label, label))
                all_scores.append(float(s))
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return all_preds, all_scores

def export_excel(df):
    """Export dataframe to Excel with ONLY display columns"""
    output = BytesIO()
    
    # Define exactly which columns to export (matching the display)
    export_cols = []
    
    # Core columns in order
    if 'project_title' in df.columns:
        export_cols.append('project_title')
    if 'client' in df.columns:
        export_cols.append('client')
    if 'project_type' in df.columns:
        export_cols.append('project_type')
    if 'predicted_label' in df.columns:
        export_cols.append('predicted_label')
    if 'confidence' in df.columns:
        export_cols.append('confidence')
    if 'prediction_status' in df.columns:
        export_cols.append('prediction_status')
    
    export_df = df[export_cols].copy()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Predictions')
        worksheet = writer.sheets['Predictions']
        for idx, col in enumerate(export_df.columns):
            max_len = max(export_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx) if idx < 26 else 'AA'].width = min(max_len, 50)
    
    return output.getvalue()

# ==================== MAIN APPLICATION ====================
def main():
    st.title("Batch Inference System")
    st.caption("Professional project classification for enterprise workflows")
    
    # Load model
    with st.spinner("Initializing model..."):
        model, tokenizer, original_id2label, consolidated_classes, device = load_model()
    
    st.caption(f"System Status: {'GPU Acceleration' if torch.cuda.is_available() else 'CPU Processing'}")
    
    # Tabs
    tab1, tab2 = st.tabs(["Batch Processing", "Configuration"])
    
    with tab1:
        st.markdown("### Data Input")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Select Excel files for batch processing",
            type=["xlsx", "xls", "xlsm"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.info(f"Files loaded: {len(uploaded_files)} • Total size: {total_size:.2f} MB")
            
            # Confidence threshold
            col1, col2 = st.columns([3, 1])
            
            with col1:
                confidence_threshold = st.slider(
                    "Minimum Confidence Level",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    key="confidence_slider"
                )
                st.write(f"Selected threshold: {confidence_threshold:.0%}")
                st.session_state.confidence_threshold = confidence_threshold
            
            with col2:
                process_button = st.button("Run Batch Inference", type="primary", use_container_width=True)
            
            # Process files when button clicked
            if process_button and not st.session_state.processing_complete:
                start_time = time.time()
                
                with st.spinner("Processing files..."):
                    df = process_files(uploaded_files)
                
                if df.empty:
                    st.error("No valid data found")
                else:
                    # Validate columns
                    if 'project_title' not in df.columns or 'client' not in df.columns:
                        st.error("Missing required columns: project_title and/or client")
                    else:
                        # Prepare data
                        df['project_title'] = df['project_title'].fillna('').astype(str).str.strip()
                        df['client'] = df.get('client', '').fillna('').astype(str).str.strip()
                        df['text'] = (df['project_title'] + ' ' + df['client']).str.strip()
                        df = df[df['text'].str.len() > 0].copy()
                        
                        # Run predictions
                        with st.spinner(f"Classifying {len(df):,} projects..."):
                            texts = df['text'].tolist()
                            predictions, scores = predict_batch(texts, model, tokenizer, original_id2label, device)
                        
                        # Add results
                        df['predicted_label'] = predictions
                        df['confidence'] = scores
                        df['needs_review'] = df['confidence'] < confidence_threshold
                        df['prediction_status'] = df['needs_review'].map({True: 'Review Required', False: 'High Confidence'})
                        
                        # Store in session state
                        st.session_state.processed_df = df
                        st.session_state.processing_complete = True
                        st.session_state.process_time = time.time() - start_time
                        
                        st.success(f"Processing completed in {st.session_state.process_time:.1f} seconds")
                        st.rerun()
            
            # Display results if processing is complete
            if st.session_state.processing_complete and st.session_state.processed_df is not None:
                df = st.session_state.processed_df
                
                # Metrics
                st.markdown("### Performance Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Projects", f"{len(df):,}")
                with col2:
                    st.metric("Processing Time", f"{st.session_state.process_time:.1f}s")
                with col3:
                    st.metric("Throughput", f"{len(df)/st.session_state.process_time:.0f} rows/s")
                with col4:
                    st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
                with col5:
                    st.metric("High Confidence", f"{(df['confidence'] >= st.session_state.confidence_threshold).mean():.0%}")
                
                # Statistics
                st.markdown("### Classification Statistics")
                stats = []
                for label in sorted(df['predicted_label'].unique()):
                    label_df = df[df['predicted_label'] == label]
                    stats.append({
                        'Project Type': label,
                        'Count': len(label_df),
                        'Distribution': f"{len(label_df)/len(df)*100:.1f}%",
                        'Confidence': f"{label_df['confidence'].mean():.1%}"
                    })
                
                st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
                
                # Chart
                st.markdown("### Distribution Analysis")
                dist = df['predicted_label'].value_counts().reset_index()
                dist.columns = ['Type', 'Count']
                
                fig = go.Figure(data=[go.Bar(
                    x=dist['Type'], y=dist['Count'], text=dist['Count'],
                    textposition='outside',
                    marker=dict(color=dist['Count'], colorscale='Blues', showscale=False)
                )])
                fig.update_layout(
                    xaxis_title="Project Type", yaxis_title="Number of Projects",
                    height=400, showlegend=False, plot_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Filters
                st.markdown("### Detailed Results")
                
                # Initialize filter variables
                search = ""
                type_filter = "All Types"
                conf_filter = "All"
                value_column = "None"
                min_value = 0
                max_value = 0
                
                # Get all numeric columns for value filtering
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Filter controls row 1
                col1, col2, col3 = st.columns([3, 2, 2])
                
                with col1:
                    search = st.text_input("Search", placeholder="Filter by project or client...", key="search")
                
                with col2:
                    type_filter = st.selectbox("Type", ["All Types"] + sorted(df['predicted_label'].unique()), key="type")
                
                with col3:
                    conf_filter = st.selectbox("Confidence", ["All", "90%+", "80%+", "70%+", "Below 70%"], key="conf")
                
                # Filter controls row 2 - Value filtering
                col4, col5, col6 = st.columns([2, 1.5, 1.5])
                
                with col4:
                    if numeric_columns:
                        value_column = st.selectbox(
                            "Filter by value column:",
                            ["None"] + numeric_columns,
                            key="value_col"
                        )
                    else:
                        st.write("No numeric columns found")
                
                with col5:
                    if value_column != "None" and value_column in df.columns:
                        min_val = float(df[value_column].min())
                        max_val = float(df[value_column].max())
                        min_value = st.number_input(
                            "Min Value",
                            min_value=min_val,
                            max_value=max_val,
                            value=min_val,
                            key="min_val"
                        )
                
                with col6:
                    if value_column != "None" and value_column in df.columns:
                        min_val = float(df[value_column].min())
                        max_val = float(df[value_column].max())
                        max_value = st.number_input(
                            "Max Value",
                            min_value=min_val,
                            max_value=max_val,
                            value=max_val,
                            key="max_val"
                        )
                
                # Apply filters
                filtered = df.copy()
                
                if search:
                    mask = (filtered['project_title'].str.contains(search, case=False, na=False) | 
                           filtered['client'].str.contains(search, case=False, na=False))
                    filtered = filtered[mask]
                
                if type_filter != "All Types":
                    filtered = filtered[filtered['predicted_label'] == type_filter]
                
                if conf_filter != "All":
                    if conf_filter == "90%+":
                        filtered = filtered[filtered['confidence'] >= 0.9]
                    elif conf_filter == "80%+":
                        filtered = filtered[filtered['confidence'] >= 0.8]
                    elif conf_filter == "70%+":
                        filtered = filtered[filtered['confidence'] >= 0.7]
                    elif conf_filter == "Below 70%":
                        filtered = filtered[filtered['confidence'] < 0.7]
                
                # Apply value filter if a column is selected
                if value_column != "None" and value_column in df.columns:
                    filtered = filtered[(filtered[value_column] >= min_value) & (filtered[value_column] <= max_value)]
                
                # Display columns
                display_cols = ['project_title', 'client']
                if 'project_type' in filtered.columns:
                    filtered['manual_input'] = filtered['project_type']
                    display_cols.append('manual_input')
                display_cols.extend(['predicted_label', 'confidence', 'prediction_status'])
                
                if 'manual_input' in filtered.columns:
                    filtered['match'] = filtered.apply(
                        lambda x: '✓' if x.get('manual_input') == x['predicted_label'] else '✗', axis=1
                    )
                    display_cols.append('match')
                
                show = filtered[display_cols].copy()
                
                # Apply color styling to dataframe
                def style_confidence(val):
                    if isinstance(val, str):
                        return ''
                    if val >= 0.9:
                        return 'background-color: #d4f4dd; color: #22c55e; font-weight: bold'
                    elif val >= 0.7:
                        return 'background-color: #fef3c7; color: #f59e0b; font-weight: bold'
                    else:
                        return 'background-color: #fee2e2; color: #ef4444; font-weight: bold'
                
                # Apply styling and format
                show_styled = show.head(500).style.applymap(style_confidence, subset=['confidence'])
                show_styled = show_styled.format({'confidence': '{:.1%}'})
                
                st.dataframe(
                    show_styled,
                    use_container_width=True, 
                    height=400,
                    column_config={
                        'project_title': st.column_config.TextColumn('Project Title', width='large'),
                        'client': st.column_config.TextColumn('Client', width='medium'),
                        'manual_input': st.column_config.TextColumn('Manual Input', width='medium'),
                        'predicted_label': st.column_config.TextColumn('Prediction', width='medium'),
                        'confidence': st.column_config.TextColumn('Confidence', width='small'),
                        'prediction_status': st.column_config.TextColumn('Status', width='medium'),
                        'match': st.column_config.TextColumn('Match', width='small')
                    }
                )
                st.caption(f"Showing {min(500, len(filtered))} of {len(filtered)} results")
                
                # Export
                st.markdown("### Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    excel_full = export_excel(df)
                    st.download_button(
                        "Export All to Excel",
                        excel_full,
                        f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    if len(filtered) < len(df):
                        excel_filtered = export_excel(filtered)
                        st.download_button(
                            f"Export Filtered ({len(filtered)} rows)",
                            excel_filtered,
                            f"filtered_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                with col3:
                    if st.button("Clear Results"):
                        st.session_state.processed_df = None
                        st.session_state.processing_complete = False
                        st.session_state.process_time = 0
                        st.rerun()
    
    with tab2:
        st.markdown("### System Configuration")
        st.info(f"""
        **Model**: {CKPT_DIR}
        **Batch Size**: {BATCH_SIZE}
        **Max Tokens**: {MAX_LENGTH}
        **Device**: {'GPU - ' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
        """)

if __name__ == "__main__":
    main()
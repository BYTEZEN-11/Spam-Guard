import streamlit as st
import pandas as pd
import tempfile
import os
import time
from pathlib import Path
from src.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_INPUT_LENGTH = 200_000

@st.cache_resource
def get_pipeline():
    return PredictionPipeline(load_models=True)

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

st.title("📧 Spam Email Classifier")
st.markdown("Classify emails as **Spam** or **Ham** (Clean) using Machine Learning.")

tab1, tab2 = st.tabs(["Single Email", "Batch MBOX Processing"])

with tab1:
    st.header("Check a Single Email")
    email_text = st.text_area(
        "Paste the email content here:",
        height=200,
        placeholder="Dear friend, I have a business proposal...",
        max_chars=MAX_INPUT_LENGTH,
    )

    if st.button("Classify Email", type="primary"):
        if email_text.strip():
            with st.spinner("Analyzing..."):
                try:
                    result = pipeline.predict_single_email(email_text)
                    prediction = result['prediction']
                    confidence = result.get('confidence')

                    if prediction == "Spam":
                        st.error(f"🚨 This email is **SPAM**")
                    else:
                        st.success(f"✅ This email is **HAM** (Safe)")

                    if confidence is not None:
                        st.info(f"Confidence Score: {confidence:.1f}%")

                except ValueError as ve:
                    st.warning(str(ve))
                except Exception as e:
                    logger_msg = str(e)
                    st.error(f"Error analyzing email. Please verify the input.")
                    try:
                        from src.utils.logger import get_logger
                        get_logger(__name__).error(f"Single-email predict failed: {logger_msg}")
                    except Exception:
                        pass
        else:
            st.warning("Please enter some text to classify.")

with tab2:
    st.header("Process MBOX File")
    uploaded_file = st.file_uploader(
        "Upload an MBOX file",
        type=['mbox', 'txt'],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        size = getattr(uploaded_file, "size", None) or len(uploaded_file.getvalue())
        if size > MAX_UPLOAD_BYTES:
            st.error(
                f"File too large ({size / (1024*1024):.1f} MB). "
                f"Maximum allowed is {MAX_UPLOAD_BYTES / (1024*1024):.0f} MB."
            )
            st.stop()

        if st.button("Process File"):
            tmp_path: str | None = None
            with st.spinner("Processing file... this may take a moment"):
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mbox')
                    try:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    finally:
                        tmp.close()

                    df = pipeline.predict_mbox_file(tmp_path)

                    spam_count = int((df['Prediction'] == 'Spam').sum())
                    ham_count = int((df['Prediction'] == 'Ham').sum())
                    error_count = int((df['Prediction'] == 'Error').sum())

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Emails", len(df))
                    col2.metric("Spam Found", spam_count, delta_color="inverse")
                    col3.metric("Ham Found", ham_count)

                    if error_count:
                        st.warning(f"{error_count} email(s) failed to classify.")

                    preview_cols = [c for c in ['Time', 'Subject', 'Prediction', 'Confidence'] if c in df.columns]
                    st.subheader("Results Preview")
                    st.dataframe(df[preview_cols].head(10))

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Results (CSV)",
                        data=csv,
                        file_name=f"predictions_{int(time.time())}.csv",
                        mime="text/csv",
                    )

                except FileNotFoundError as fnf:
                    st.error(str(fnf))
                except ValueError as ve:
                    st.warning(str(ve))
                except Exception as e:
                    st.error("Error processing file. See logs for details.")
                    try:
                        from src.utils.logger import get_logger
                        get_logger(__name__).error(f"MBOX processing failed: {e}")
                    except Exception:
                        pass
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        for _ in range(2):
                            try:
                                os.unlink(tmp_path)
                                break
                            except OSError:
                                time.sleep(0.2)

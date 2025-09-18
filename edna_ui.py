# edna_ui.py
import streamlit as st
import io
from matplotlib import pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from edna_predict import predict_sequences, fetch_annotation

st.set_page_config(page_title="🌊 eDNA Taxonomy Predictor", layout="wide")
st.title("🌊 eDNA Taxonomy Predictor")

# Custom CSS for larger fonts and highlights
st.markdown("""
<style>
.big-font { font-size:20px !important; color: #0d6efd;}
.highlight { font-weight: bold; color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# Multiple file upload
uploaded_files = st.file_uploader(
    "Upload FASTA/CSV/TXT",
    type=["fasta","fa","csv","txt"],
    accept_multiple_files=True
)

top_n = st.slider("Top N predictions", 1, 20, 5)

if uploaded_files and st.button("Predict"):
    st.info("Running predictions... This may take a few seconds.")
    all_results = {}
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    y_pos = height - 50

    for file in uploaded_files:
        st.markdown(f"## 📄 {file.name}")
        temp_path = file.name
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        results = predict_sequences(temp_path, top_n)
        all_results[file.name] = results

        for seq_id, seq_res in results.items():
            st.markdown(f"### 🧬 Sequence: {seq_id}", unsafe_allow_html=True)

            # Sort by confidence
            seq_res_sorted = dict(sorted(seq_res.items(), key=lambda x: x[1], reverse=True))
            
            # Display top prediction as a metric
            top_label, top_conf = list(seq_res_sorted.items())[0]
            st.metric(label="Top Prediction", value=f"{top_label}", delta=f"{top_conf:.2f}%")

            # Columns for pie chart and other predictions
            col1, col2 = st.columns([1,2])

            with col1:
                filtered_res = {k: v for k,v in seq_res_sorted.items() if v>=0.01}
                if filtered_res:
                    labels = list(filtered_res.keys())
                    scores = list(filtered_res.values())
                    fig, ax = plt.subplots()
                    ax.pie(scores, labels=labels, autopct="%1.1f%%", startangle=140)
                    ax.set_title("Prediction Distribution", fontsize=14)
                    st.pyplot(fig)

            with col2:
                for label, conf in seq_res_sorted.items():
                    if conf >= 0.01:
                        st.markdown(f"<p class='big-font'>🔹 {label} : {conf:.2f}%</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='color:gray;'>🔹 {label} : {conf:.6f}%</p>", unsafe_allow_html=True)

            # Collapsible annotation
            annotation = fetch_annotation(seq_id)
            with st.expander("📝 View Annotation"):
                st.write(annotation)

            # --- Write to PDF ---
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_pos, f"Sequence ID: {seq_id}")
            y_pos -= 16
            c.setFont("Helvetica", 10)
            c.drawString(50, y_pos, f"Annotation: {annotation}")
            y_pos -= 14
            for label, conf in seq_res_sorted.items():
                c.drawString(60, y_pos, f"- {label}: {conf:.2f}%")
                y_pos -= 12

            # Pie chart for PDF
            if filtered_res:
                plt.figure(figsize=(3,3))
                plt.pie(list(filtered_res.values()), labels=list(filtered_res.keys()), autopct="%1.1f%%", startangle=140)
                chart_buf = io.BytesIO()
                plt.savefig(chart_buf, format='PNG', bbox_inches="tight")
                plt.close()
                chart_buf.seek(0)
                c.drawImage(ImageReader(chart_buf), 50, y_pos-200, width=250, height=200)
                y_pos -= 220

            if y_pos < 100:
                c.showPage()
                y_pos = height - 50

    c.save()
    pdf_buffer.seek(0)

    # Download button
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="prediction_report.pdf",
        mime="application/pdf"
    )

    st.success("✅ Predictions complete! PDF ready for download.")

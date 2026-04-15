from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import os


def sanitize_text(t):
    if not isinstance(t, str):
        return str(t)
    return t.encode('latin-1', 'replace').decode('latin-1')

def generate_report(ticker, fusion_result, headlines, explanation):

    # Generate Chart
    df = yf.download(ticker, period="6mo", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Close"], color="#2B5B84", linewidth=2.5)
    plt.title(f"{ticker} 6-Month Asset Pricing Trend", fontsize=15, fontweight='bold', color="#112233")
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    chart_path = f"{ticker}_trend.png"
    plt.savefig(chart_path, dpi=120)
    plt.close()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(30, 60, 90)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_y(15)
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, txt=f"{ticker} Intelligence Report", ln=True, align="C")

    pdf.ln(20)
    pdf.set_text_color(0, 0, 0)
    
    # Render Chart
    pdf.image(chart_path, x=15, w=180)
    os.remove(chart_path)

    pdf.ln(5)

    # Core Metrics Table
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(30, 60, 90)
    pdf.cell(0, 10, txt="Algorithmic Consensus & Metrics:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(40, 40, 40)
    
    metrics = [
        f"Target Designation:  {fusion_result['decision']} (Conf: {fusion_result['confidence']:.3f})",
        f"Core Network Signal: {fusion_result['rl_signal']:.4f}",
        f"NLP Market Sentiment: {fusion_result['sentiment']:.3f}",
        f"Current Price Regime: {fusion_result['regime']}"
    ]
    for m in metrics:
        pdf.cell(0, 7, txt=f"  > {m}", ln=True)

    pdf.ln(5)

    # Reasoning Paragraph
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(30, 60, 90)
    pdf.cell(0, 10, txt="System Architecture Reasoning:", ln=True)
    pdf.set_font("Arial", 'I', 11)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(0, 6, sanitize_text(explanation))

    pdf.ln(5)

    # Headlines Section
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(30, 60, 90)
    pdf.cell(0, 10, txt="Contextual NLP Corpus Evaluated:", ln=True)

    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(50, 50, 50)

    if headlines:
        for h in headlines:
            pdf.multi_cell(0, 6, sanitize_text(f"- {h}"))
            pdf.ln(1)
    else:
        pdf.cell(0, 8, txt="  No relevant headlines fetched safely.", ln=True)

    # Save format
    filename = f"{ticker}_report.pdf"
    pdf.output(filename)

    return filename
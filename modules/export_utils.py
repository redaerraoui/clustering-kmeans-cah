"""Export des résultats en PDF et en ZIP, entièrement en mémoire."""
import io
import zipfile

from fpdf import FPDF


def figure_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def export_pdf(title, log_lines, png_bytes):
    """Renvoie un PDF (bytes) contenant le titre, le journal et la figure."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    for line in log_lines:
        pdf.multi_cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    if png_bytes:
        pdf.image(io.BytesIO(png_bytes), x=10, w=190)
    return bytes(pdf.output())


def export_zip(log_txt, df, png_bytes, base_name):
    """Renvoie une archive ZIP (bytes) avec le journal, les affectations en CSV et la figure."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("log.txt", log_txt)
        z.writestr(f"{base_name}_clusters.csv", df.to_csv(index=False))
        if png_bytes:
            z.writestr(f"{base_name}.png", png_bytes)
    return buf.getvalue()

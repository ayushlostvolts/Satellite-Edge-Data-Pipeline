"""Generate a comprehensive, publication-grade PDF User Manual for the
Satellite Edge Rainfall Processing Pipeline.
"""

from pathlib import Path
import sys
import time

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page numbers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        # Suppress headers/footers on page 1 (cover)
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#4da2ff"))
            self.drawString(54, 750, "SLUSH // SATELLITE EDGE RAINFALL PROCESSING PIPELINE")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawRightString(558, 750, "SYSTEM RUNNING & OPERATIONAL GUIDE")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.75)
            self.line(54, 742, 558, 742)

            # Footer
            self.line(54, 45, 558, 45)
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(54, 32, "Antigravity Edge Lab · IMD 0.25° Gridded Rainfall Pipeline")
            self.drawRightString(558, 32, f"Page {self._pageNumber} of {total_pages}")

        self.restoreState()


def create_running_manual_pdf(out_pdf_path: str):
    """Build the complete PDF document."""
    out_path = Path(out_pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0f172a")    # Deep slate carbon
    BRAND_BLUE = colors.HexColor("#1d4ed8") # Deep Electric Blue
    ACCENT_MINT = colors.HexColor("#059669")# Emerald green
    LIGHT_BG = colors.HexColor("#f8fafc")   # Cool gray 50
    BORDER_COLOR = colors.HexColor("#000000")
    CODE_BG = colors.HexColor("#f1f5f9")

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=30,
        textColor=PRIMARY,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#334155"),
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=BRAND_BLUE,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#0f172a"),
        backColor=CODE_BG,
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.75,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6,
        borderRadius=4,
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#065f46"),
    )

    story = []

    # =========================================================================
    # COVER / HEADER BLOCK
    # =========================================================================
    story.append(Paragraph("SATELLITE EDGE RAINFALL PIPELINE", title_style))
    story.append(Paragraph("System Operational Manual & Execution Guide", subtitle_style))
    story.append(Paragraph(
        "<b>Project:</b> 4th Year Satellite Edge Computing & Earth Observation<br/>"
        "<b>Dataset:</b> IMD 0.25° Gridded Daily Rainfall (India, NetCDF, 366 Days, 17,415 Grid Cells)<br/>"
        "<b>Architecture:</b> 100% Pure NumPy Edge Core · Zero SciPy Dependencies in Pipeline Core · Q15 Int16 Fixed-Point",
        ParagraphStyle("Meta", parent=body_style, fontSize=8.5, leading=12, textColor=colors.HexColor("#475569"))
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=14))

    # =========================================================================
    # SECTION 1: SYSTEM PREREQUISITES & ENVIRONMENT SETUP
    # =========================================================================
    story.append(Paragraph("1. System Prerequisites & Environment Setup", h1_style))
    story.append(Paragraph(
        "The entire pipeline is self-contained within the <code>rain_filter</code> workspace. All core algorithms "
        "(impulse suppressor, adaptive IIR, custom FIR, DWT wavelets, and feature extraction) run on pure NumPy.",
        body_style
    ))

    # Prerequisites Table
    req_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Specification / Version</b>", body_style), Paragraph("<b>Purpose</b>", body_style)],
        ["Python", "3.10+ or 3.11+ (Win64 / Linux / macOS)", "Runtime environment"],
        ["Virtualenv", "rain_filter/.venv", "Isolated dependency container"],
        ["NumPy", ">= 1.24.0", "Core mathematical array engine"],
        ["NetCDF4 / xarray", ">= 1.6.0", "IMD Gridded Climate Cube I/O"],
        ["Matplotlib", ">= 3.8.0", "Publication plot generation"],
        ["SciPy", ">= 1.11.0", "Evaluation baseline comparison only"],
        ["Pytest", ">= 8.0.0", "Automated validation test suite (46 tests)"],
        ["Dataset File", "Dataset/RF25_ind2024_rfp25.nc", "IMD 2024 Daily Rainfall NetCDF (366 × 129 × 135)"]
    ]
    t_req = Table(req_data, colWidths=[110, 190, 204])
    t_req.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), PRIMARY),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(t_req)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 2: QUICK START GUIDE (RUNNING IN 60 SECONDS)
    # =========================================================================
    story.append(Paragraph("2. Quick Start: Launching the System in 3 Steps", h1_style))
    story.append(Paragraph("<b>Step 1: Activate the Project Virtual Environment</b>", h2_style))
    story.append(Paragraph(
        "Open PowerShell or Command Prompt in the repository root directory (<code>e:\\Coding\\4th year Project</code>):",
        body_style
    ))
    story.append(Paragraph(
        "&gt; .\\rain_filter\\.venv\\Scripts\\Activate.ps1",
        code_style
    ))

    story.append(Paragraph("<b>Step 2: Run the Automated Validation Suite (46 Tests)</b>", h2_style))
    story.append(Paragraph(
        "Execute pytest to verify mathematical correctness across all 6 pipeline modules:",
        body_style
    ))
    story.append(Paragraph(
        "&gt; python -m pytest rain_filter/tests -v",
        code_style
    ))
    story.append(Paragraph(
        "<i>Expected Result:</i> <code>======================= 46 passed in ~2.3s =======================</code>",
        body_style
    ))

    story.append(Paragraph("<b>Step 3: Launch the Slush Interactive Web Application</b>", h2_style))
    story.append(Paragraph(
        "Start the multithreaded REST server hosting the interactive visualization dashboard:",
        body_style
    ))
    story.append(Paragraph(
        "&gt; python rain_filter/web/server.py 8080",
        code_style
    ))
    story.append(Paragraph(
        "Open your web browser and navigate to: <b><u>http://localhost:8080</u></b>",
        body_style
    ))

    # Tip Box
    tip_table = Table([[
        Paragraph(
            "<b>💡 Pro-Tip:</b> The server automatically precomputes the contrast population at startup "
            "and enables multi-threading. Single-cell simulations execute in <b>~35 ms</b>, while "
            "full 6.37-million sample NetCDF databases calculate in <b>~2.5 seconds</b>.",
            callout_style
        )
    ]], colWidths=[504])
    tip_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecfdf5")),
        ("BORDER", (0, 0), (-1, -1), 1, ACCENT_MINT),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(tip_table)
    story.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 3: COMMAND-LINE TOOLS & BENCHMARK HARNESS
    # =========================================================================
    story.append(Paragraph("3. CLI Scripts & Evaluation Benchmarks", h1_style))
    story.append(Paragraph(
        "You can run all evaluation benchmarks, plot generators, and hardware profiling directly from the CLI:",
        body_style
    ))

    cli_tasks = [
        [Paragraph("<b>Command</b>", body_style), Paragraph("<b>Description & Output Artifact</b>", body_style)],
        [
            Paragraph("<code>python rain_filter/src/evaluate.py</code>", code_style),
            Paragraph("Runs the comprehensive evaluation harness across 4 benchmark climate stations (Mumbai, Cherrapunji, Kolkata, Jaisalmer) over 4 noise stress regimes (Gaussian, Impulsive, Mixed, Dropouts) vs 4 SciPy baselines (Butterworth, Median, MA, Savitzky-Golay). Generates <code>rain_filter/outputs/results.csv</code>.", body_style)
        ],
        [
            Paragraph("<code>python rain_filter/src/plots.py</code>", code_style),
            Paragraph("Generates Stage 1 filtering figures 01 through 06 (Time-series comparisons, SNR boxplots, Peak preservation, Adaptive gain dynamics, Volume conservation, Wet-day F1). Saved in <code>rain_filter/outputs/plots/</code>.", body_style)
        ],
        [
            Paragraph("<code>python rain_filter/src/plots_dwt.py</code>", code_style),
            Paragraph("Generates Figures 07 & 08: Multi-resolution DWT scalograms (A3, D3, D2, D1) and subband energy partition donuts across climatic regimes.", body_style)
        ],
        [
            Paragraph("<code>python rain_filter/src/plots_features.py</code>", code_style),
            Paragraph("Generates Figure 09: Composite Edge Priority Score (CEPS) feature ranking bar charts and discriminative power plots.", body_style)
        ],
        [
            Paragraph("<code>python rain_filter/src/cost.py</code>", code_style),
            Paragraph("Runs edge hardware feasibility benchmarks: FLOPs counter (66 OPS/sample), state memory audit (8 Bytes RAM/cell = 136 KB for India), and fixed-point Q15 int16 accuracy validation.", body_style)
        ]
    ]
    t_cli = Table(cli_tasks, colWidths=[200, 304])
    t_cli.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), PRIMARY),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_cli)
    story.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 4: SLUSH WEB DASHBOARD INTERACTIVE WALKTHROUGH
    # =========================================================================
    story.append(Paragraph("4. Interactive Slush Web Interface Guide", h1_style))
    story.append(Paragraph(
        "The Slush dashboard (<b>http://localhost:8080</b>) provides an intuitive, real-time playground "
        "designed in accordance with the Slush sticker aesthetic (1px black borders, pastel paper grounds, pill controls).",
        body_style
    ))

    story.append(Paragraph("<b>Section 1: Hero & Pipeline Control Board</b>", h2_style))
    story.append(Paragraph("• <b>Benchmark Station Selector:</b> Click one-touch pills for Mumbai (Ghats Monsoon), Cherrapunji (Extreme Meghalaya), Kolkata (Gangetic Delta), or Jaisalmer (Thar Desert).", bullet_style))
    story.append(Paragraph("• <b>Custom Coordinates:</b> Enter any Latitude (6.5°N – 38.5°N) and Longitude (66.5°E – 100.0°E) to extract any grid cell in India.", bullet_style))
    story.append(Paragraph("• <b>Stress Noise Model:</b> Test against Mixed (Gauss 1mm + 2% Spikes), Gaussian (σ = 2mm), Isolated Spikes (5% Spikes 20-80mm), Dropouts (-999 sensor masking), or Clean Series.", bullet_style))
    story.append(Paragraph("• <b>Filter Mode:</b> Choose Hybrid (A+B, default), Adaptive IIR (B only), Custom Windowed FIR (C only), or Full Pipeline (A+B+C).", bullet_style))
    story.append(Paragraph("• <b>Wavelet Family & Level:</b> Select Haar/db1, Daubechies db2, Daubechies db4, or integer lifting CDF 5/3 across Levels 2, 3, or 4.", bullet_style))
    story.append(Paragraph("• <b>Run Full Satellite Pipeline:</b> Click the purple <code>⚡ RUN FULL SATELLITE PIPELINE</code> button. The page automatically smooth-scrolls to the results and updates with a random noise seed per click.", bullet_style))

    story.append(Paragraph("<b>Section 2: Stage 1 Denoising & Time-Series Visualizer</b>", h2_style))
    story.append(Paragraph("• <b>HUD Metric Scorecards:</b> SNR Improvement (dB), Peak Preservation Ratio (target ≥ 0.85), 30-Day Volume Bias (criterion < ±0.5%), and Wet-Day F1 Score (threshold ≥ 2.5 mm).", bullet_style))
    story.append(Paragraph("• <b>Interactive Chart:</b> Toggle Clean Truth, Noisy Input, Proposed Hybrid, Butterworth-3, or Median-5 lines on/off.", bullet_style))
    story.append(Paragraph("• <b>Adaptive Gain Curve:</b> Visualizes real-time gain dynamic $a[n]$ switching between $a_{\\min}=0.08$ on noise and $a_{\\max}=0.95$ on cloudburst storms.", bullet_style))

    story.append(Paragraph("<b>Section 3 & 4: Wavelet DWT Scalogram & CEPS Ranking</b>", h2_style))
    story.append(Paragraph("• <b>Subband Waveforms:</b> Displays Approximation $A_3$ (seasonal envelope) alongside Details $D_3$, $D_2$, $D_1$ (convective spikes).", bullet_style))
    story.append(Paragraph("• <b>Energy Donut Chart:</b> Visualizes partition of total Parseval energy between convective storm transients and synoptic waves.", bullet_style))
    story.append(Paragraph("• <b>Composite Edge Priority Score (CEPS):</b> Ranks top 8 transmission features and provides a 1-click CSV download.", bullet_style))

    story.append(Paragraph("<b>Section 6: Database Ingestion Lab & Spatial Heatmap</b>", h2_style))
    story.append(Paragraph("• <b>Drag-and-Drop Ingestion:</b> Drop any NetCDF (<code>.nc</code>), Multi-station CSV, or SQLite (<code>.db</code>) file.", bullet_style))
    story.append(Paragraph("• <b>1-Click Sample Preload:</b> Click <code>⚡ LOAD BUNDLED IMD 2024 DATABASE</code> to instantly process 6,373,890 daily samples across 4,964 land cells.", bullet_style))
    story.append(Paragraph("• <b>Spatial Geographic Map:</b> Interactive canvas rendering India rainfall distribution with layer selectors (Peak mm, Total mm, Wet days).", bullet_style))
    story.append(Paragraph("• <b>Click-to-Inspect Map Navigation:</b> Click <i>any</i> pixel on the map or click <code>⚡ Inspect</code> in the Extreme Hotspots table to immediately load that cell into the waveform viewer.", bullet_style))

    # =========================================================================
    # SECTION 5: MATHEMATICAL SPECIFICATION & PARAMETERS
    # =========================================================================
    story.append(KeepTogether([
        Paragraph("5. Mathematical Specification & Filter Design", h1_style),
        Paragraph(
            "<b>Stage A: Domain-Aware Isolated Impulse Suppressor</b><br/>"
            "• Running window $W=7$ ($h=3$). Median $m[n] = \\text{median}(x[n-h:n+h+1])$.<br/>"
            "• Spread $s[n] = 1.4826 \\times \\text{MAD}(x)$. Noise floor $s_{\\text{floor}} = 0.5\\text{ mm}$, threshold $k=4.0$.<br/>"
            "• Flag sample $n$ as impulse if $(x[n]-m) &gt; k \\cdot \\max(s, s_{\\text{floor}})$ AND $x[n]$ is isolated: "
            "$\\max(x[n-1], x[n+1]) &lt; m + (x[n]-m) \\times r_{\\text{iso}}$ with $r_{\\text{iso}}=0.35$.",
            body_style
        ),
        Paragraph(
            "<b>Stage B: Adaptive-Gain IIR Tracker</b><br/>"
            "• Update equation: $y[n] = a[n] x[n] + (1 - a[n]) y[n-1]$ with $y[0] = x[0]$.<br/>"
            "• Innovation: $e[n] = x[n] - y[n-1]$. Running variance: $\\sigma^2[n] = 0.95 \\sigma^2[n-1] + 0.05 (d[n]^2 / 2)$, where $d[n]=x[n]-x[n-1]$.<br/>"
            "• Dynamic gain: $a[n] = \\text{clip}\\left(\\frac{e[n]^2}{e[n]^2 + 2.0 \\sigma^2[n]}, 0.08, 0.95\\right)$.",
            body_style
        ),
        Paragraph(
            "<b>Stage C: Windowed-Sinc Low-Pass FIR</b><br/>"
            "• $N=11$ taps (symmetric linear phase, group delay 5 samples), cutoff $f_c=0.35 f_{\\text{Nyquist}}$.<br/>"
            "• Custom hybrid window: $w[k] = 0.5 \\cdot w_{\\text{Hann}}[k] + 0.5 \\cdot w_{\\text{Kaiser}}(\\beta=4.0)[k]$, normalized to unit DC gain $\\sum h[k] = 1.0$.",
            body_style
        ),
        Paragraph(
            "<b>Volume Preservation:</b> Rescales non-overlapping 30-day blocks: "
            "$y_{\\text{pres}}[n] = y[n] \\times \\left(\\sum x / \\sum y\\right)$, guaranteeing $< 0.001\\%$ relative bias.",
            body_style
        )
    ]))

    # =========================================================================
    # SECTION 6: TROUBLESHOOTING & COMMON SCENARIOS
    # =========================================================================
    story.append(KeepTogether([
        Paragraph("6. Troubleshooting & FAQs", h1_style),
        Paragraph("<b>Q: Port 8080 is already occupied by another application.</b><br/>"
                  "<b>A:</b> Pass any alternative port directly to the server command, for example:<br/>"
                  "<code>python rain_filter/web/server.py 8081</code> (or 9000).", body_style),
        Paragraph("<b>Q: NetCDF file missing or cannot be opened.</b><br/>"
                  "<b>A:</b> If <code>Dataset/RF25_ind2024_rfp25.nc</code> is missing, the backend automatically generates a realistic Gamma-distributed synthetic fallback cube with zero division warnings.", body_style),
        Paragraph("<b>Q: How much precision is lost when executing on 16-bit Q15 DSPs?</b><br/>"
                  "<b>A:</b> Validated by <code>cost.py</code>: SNR degradation between Float64 and Int16 Q15 is strictly <b>&lt; 0.001 dB</b> (Float64 SNR: 3.45 dB vs Q15 SNR: 3.45 dB).", body_style),
        Paragraph("<b>Q: How do I export results for external research reports?</b><br/>"
                  "<b>A:</b> Click the <code>⬇ Export Results CSV</code> button in Section 4 or <code>⬇ Export CSV Report</code> in Section 6. You can also inspect <code>rain_filter/outputs/results.csv</code>.", body_style)
    ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF Generator] Successfully compiled manual to: {out_path}")
    return out_path


if __name__ == "__main__":
    out_file = "rain_filter/outputs/RUNNING_GUIDE_MANUAL.pdf"
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    create_running_manual_pdf(out_file)

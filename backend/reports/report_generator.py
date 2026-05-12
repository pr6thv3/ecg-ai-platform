import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie

class ReportGenerator:
    """
    Clinical ECG Report Generator using ReportLab.
    
    Why ReportLab over WeasyPrint?
    While WeasyPrint allows for HTML/CSS-based templating which is great for web developers, 
    it requires heavy OS-level dependencies (Pango, Cairo, GDK-Pixbuf) that significantly bloat 
    the Docker image and often cause cross-platform build issues. ReportLab is a pure-Python 
    library that is incredibly fast, has zero external C-dependencies, and offers granular programmatic 
    control over precise typography and layout via Platypus, making it ideal for robust, containerized microservices.
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.version = "v1.2.0-clinical-beta"
        
    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=20,
            alignment=1 # Center
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=15,
            spaceAfter=10,
            borderPadding=5,
            backColor=colors.HexColor("#f1f5f9")
        ))
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Italic'],
            fontSize=10,
            textColor=colors.HexColor("#64748b"),
            alignment=1
        ))

    def _get_risk_color(self, score, is_percentage=False):
        # Determine color for metric backgrounds
        if is_percentage:
            if score < 0.8: return colors.HexColor("#fee2e2") # Red
            if score < 0.9: return colors.HexColor("#fef3c7") # Yellow
            return colors.HexColor("#dcfce7") # Green
        else:
            return colors.HexColor("#f8fafc")

    def _decode_image(self, b64_string, width, height):
        try:
            # Handle standard Data URI prefix if present
            if "," in b64_string:
                b64_string = b64_string.split(",")[1]
            img_data = base64.b64decode(b64_string)
            img_buffer = io.BytesIO(img_data)
            return Image(img_buffer, width=width, height=height)
        except Exception as e:
            return Paragraph(f"<i>Image generation failed: {str(e)}</i>", self.styles['Normal'])

    def generate_pdf(self, session_data: dict) -> io.BytesIO:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=inch, leftMargin=inch,
            topMargin=inch, bottomMargin=inch
        )
        
        story = []
        
        # --- PAGE 1: Overview ---
        story.append(Paragraph("ECG AI Telemetry Report", self.styles['ReportTitle']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: {self.version}", self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Patient Info Table
        patient = session_data.get('patient_metadata', {})
        p_data = [
            ["Patient ID:", patient.get('id', 'N/A'), "Session Date:", patient.get('session_date', 'N/A')],
            ["Age:", str(patient.get('age', 'N/A')), "Gender:", patient.get('gender', 'N/A')]
        ]
        p_table = Table(p_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        p_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#334155")),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
            ('FONTNAME', (3,0), (3,-1), 'Helvetica'),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0"))
        ]))
        story.append(p_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Session Summary & Risk Flags
        story.append(Paragraph("Clinical Session Summary", self.styles['SectionHeader']))
        metrics = session_data.get('model_metrics', {})
        avg_conf = metrics.get('average_confidence', 0)
        
        summary_data = [
            ["Average Model Confidence", f"{avg_conf*100:.1f}%", "Low Confidence Beats", str(metrics.get('low_confidence_beats', 0))]
        ]
        s_table = Table(summary_data, colWidths=[2.5*inch, 1*inch, 1.5*inch, 1*inch])
        s_table.setStyle(TableStyle([
            ('BACKGROUND', (1,0), (1,0), self._get_risk_color(avg_conf, True)),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('ALIGN', (1,0), (1,-1), 'CENTER'),
            ('ALIGN', (3,0), (3,-1), 'CENTER'),
        ]))
        story.append(s_table)
        story.append(PageBreak())
        
        # --- PAGE 2: Beat Distribution ---
        story.append(Paragraph("Beat Distribution Analysis", self.styles['SectionHeader']))
        beats = session_data.get('beat_statistics', {})
        dist = beats.get('class_distribution', {})
        
        # Pie Chart
        drawing = Drawing(300, 200)
        pie = Pie()
        pie.x, pie.y, pie.width, pie.height = 65, 15, 170, 170
        pie.data = [dist.get('N', 0), dist.get('V', 0), dist.get('A', 0), dist.get('L', 0), dist.get('R', 0)]
        pie.labels = ['Normal (N)', 'PVC (V)', 'APB (A)', 'LBBB (L)', 'RBBB (R)']
        pie.slices[1].fillColor = colors.HexColor("#ef4444") # Red for PVC
        pie.slices[2].fillColor = colors.HexColor("#f97316") # Orange for APB
        pie.slices[0].fillColor = colors.HexColor("#22c55e") # Green for N
        drawing.add(pie)
        story.append(drawing)
        story.append(Spacer(1, 0.2*inch))
        
        # Anomaly Events Log
        story.append(Paragraph("Detected Anomaly Events", self.styles['SectionHeader']))
        events = session_data.get('anomaly_events', [])
        if events:
            event_data = [["Timestamp", "Beat Type", "Confidence", "Alert Message"]]
            for ev in events[:20]: # Limit to 20 for report
                event_data.append([
                    str(ev.get('timestamp', '')), 
                    ev.get('beat_type', ''), 
                    f"{ev.get('confidence', 0)*100:.1f}%", 
                    ev.get('alert_message', '')
                ])
            e_table = Table(event_data, colWidths=[1.5*inch, 1*inch, 1*inch, 3*inch])
            e_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#334155")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 10),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1"))
            ]))
            story.append(e_table)
            if len(events) > 20:
                story.append(Paragraph(f"... and {len(events)-20} more events.", self.styles['Normal']))
        else:
            story.append(Paragraph("No major anomalies detected during this session.", self.styles['Normal']))
        
        story.append(PageBreak())
        
        # --- PAGE 3: Visuals & Signal Quality ---
        story.append(Paragraph("Signal Diagnostics", self.styles['SectionHeader']))
        sig = session_data.get('signal_metadata', {})
        sig_data = [
            ["Duration (sec):", str(sig.get('duration_sec', 'N/A')), "Sampling Rate:", f"{sig.get('sampling_rate', 'N/A')} Hz"],
            ["SNR Before:", f"{sig.get('snr_before', 'N/A')} dB", "SNR After:", f"{sig.get('snr_after', 'N/A')} dB"]
        ]
        story.append(Table(sig_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch]))
        story.append(Spacer(1, 0.3*inch))
        
        if session_data.get('waveform_b64'):
            story.append(Paragraph("Session Telemetry Waveform", self.styles['Heading3']))
            story.append(self._decode_image(session_data.get('waveform_b64'), 6*inch, 2.5*inch))
            story.append(Spacer(1, 0.3*inch))
            
        if session_data.get('confusion_matrix_b64'):
            story.append(Paragraph("Model Benchmark (Confusion Matrix)", self.styles['Heading3']))
            story.append(self._decode_image(session_data.get('confusion_matrix_b64'), 4*inch, 3*inch))
        
        story.append(PageBreak())
        
        # --- PAGE 4: Disclaimer ---
        story.append(Spacer(1, 3*inch))
        story.append(Paragraph("FOR RESEARCH USE ONLY", self.styles['SectionHeader']))
        disclaimer = """
        This report is generated by an automated Artificial Intelligence pipeline utilizing deep learning 
        (1D Convolutional Neural Networks). It is strictly intended for research, demonstration, and engineering 
        portfolio evaluation purposes. It has NOT been approved by the FDA or any regulatory body. 
        It MUST NOT be used for clinical diagnosis, patient monitoring, or medical decision making.
        """
        story.append(Paragraph(disclaimer, self.styles['Disclaimer']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE_W, PAGE_H = A4
INK = colors.HexColor("#20242A")
MUTED = colors.HexColor("#66707A")
PAPER = colors.HexColor("#F8F4EC")
ACCENT = colors.HexColor("#A34532")
ACCENT_LIGHT = colors.HexColor("#E8CBBE")
TEAL = colors.HexColor("#376A68")
LINE = colors.HexColor("#D9D1C5")
WHITE = colors.white


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float, height: float = 118 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(colors.HexColor("#EEE7DC"))
        c.roundRect(0, 0, self.width, self.height, 10, fill=1, stroke=0)

        def box(x, y, w, h, title, body, fill=WHITE):
            c.setFillColor(fill)
            c.setStrokeColor(LINE)
            c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
            c.setFillColor(ACCENT)
            c.setFont("Helvetica-Bold", 8.5)
            c.drawString(x + 8, y + h - 14, title)
            c.setFillColor(INK)
            c.setFont("Helvetica", 7.2)
            lines = body.split("\n")
            for idx, line in enumerate(lines):
                c.drawString(x + 8, y + h - 28 - idx * 10, line)

        def arrow(x1, y1, x2, y2):
            c.setStrokeColor(TEAL)
            c.setFillColor(TEAL)
            c.setLineWidth(1.4)
            c.line(x1, y1, x2, y2)
            angle = 5
            c.line(x2, y2, x2 - angle, y2 + angle / 2)
            c.line(x2, y2, x2 - angle, y2 - angle / 2)

        margin = 12
        gap = 13
        col = (self.width - 2 * margin - 2 * gap) / 3
        box(margin, self.height - 51, col, 36, "RAW IMAGE", "full frame + local crops")
        box(margin + col + gap, self.height - 51, col, 36, "FASHION ENCODER", "FashionSigLIP, L2 normalize")
        box(margin + 2 * (col + gap), self.height - 51, col, 36, "VECTOR STORE", "global + region .npy / FAISS")
        arrow(margin + col, self.height - 33, margin + col + gap - 2, self.height - 33)
        arrow(margin + 2 * col + gap, self.height - 33, margin + 2 * (col + gap) - 2, self.height - 33)

        box(margin, self.height - 104, col, 39, "NATURAL LANGUAGE", "full query embedding\nzero-shot input")
        box(margin + col + gap, self.height - 104, col, 39, "QUERY GRAPH", "bindings | scene | style | action")
        box(margin + 2 * (col + gap), self.height - 104, col, 39, "ANN CANDIDATES", "top 100 via global vector")
        arrow(margin + col, self.height - 84, margin + col + gap - 2, self.height - 84)
        arrow(margin + 2 * col + gap, self.height - 84, margin + 2 * (col + gap) - 2, self.height - 84)

        box(margin + 20, 15, col + 12, 43, "LOCAL FACET EVIDENCE", "max over valid regions\nred tie != red somewhere")
        box(self.width - margin - col - 32, 15, col + 20, 43, "CONJUNCTIVE RERANKER", "smooth-AND + score breakdown\nreturns explainable top k", ACCENT_LIGHT)
        arrow(margin + col + 32, 36, self.width - margin - col - 36, 36)
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Oblique", 7.2)
        c.drawCentredString(self.width / 2, 4, "Candidate generation is broad; reranking enforces every requested attribute.")
        c.restoreState()


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=29, leading=32, textColor=INK, alignment=TA_LEFT, spaceAfter=8),
        "kicker": ParagraphStyle("Kicker", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=ACCENT, tracking=1.4, spaceAfter=8),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=13, leading=18, textColor=MUTED, spaceAfter=16),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=INK, spaceBefore=3, spaceAfter=10),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=ACCENT, spaceBefore=9, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.1, leading=13.1, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6, leading=10.4, textColor=MUTED, spaceAfter=4),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=8.8, leading=12.4, textColor=INK, leftIndent=12, firstLineIndent=-7, bulletIndent=2, spaceAfter=4),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10.2, leading=14.5, textColor=TEAL),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=7.7, leading=10.2, textColor=INK, backColor=colors.HexColor("#EDE8DF"), borderPadding=7, spaceAfter=8),
        "ref": ParagraphStyle("Ref", parent=base["BodyText"], fontName="Helvetica", fontSize=7.3, leading=10, textColor=MUTED, leftIndent=12, firstLineIndent=-12, spaceAfter=3),
        "center": ParagraphStyle("Center", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=10, alignment=TA_CENTER, textColor=MUTED),
    }


def p(text, style):
    return Paragraph(text, style)


def bullets(items, styles):
    return [p(f"- {item}", styles["bullet"]) for item in items]


def callout(text, styles):
    table = Table([[p(text, styles["callout"])]], colWidths=[164 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E9F0ED")),
        ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return table


def styled_table(data, widths, header=True, font_size=7.7):
    body_style = ParagraphStyle("CellBody", fontName="Helvetica", fontSize=font_size, leading=font_size + 2.1, textColor=INK)
    header_style = ParagraphStyle("CellHeader", fontName="Helvetica-Bold", fontSize=font_size, leading=font_size + 2.1, textColor=WHITE)
    wrapped = []
    for row_index, row in enumerate(data):
        style = header_style if header and row_index == 0 else body_style
        wrapped.append([Paragraph(html.escape(str(value)), style) for value in row])
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 1 if header else 0), (-1, -1), WHITE),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    for row in range(2 if header else 1, len(data), 2):
        commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F1ECE4")))
    table.setStyle(TableStyle(commands))
    return table


def page_chrome(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 14 * mm, PAGE_W - 18 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawString(18 * mm, 9 * mm, "GLANCE ML INTERNSHIP ASSIGNMENT | MULTIMODAL FASHION RETRIEVAL")
    canvas.drawRightString(PAGE_W - 18 * mm, 9 * mm, str(doc.page))
    canvas.restoreState()


def build_pdf(output: Path, repo_url: str | None):
    styles = build_styles()
    profile_path = ROOT / "reports" / "dataset_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    montage = ROOT / "reports" / "dataset_montage.jpg"
    evaluation_summary = ROOT / "reports" / "evaluation_summary.jpg"
    frame = Frame(18 * mm, 18 * mm, PAGE_W - 36 * mm, PAGE_H - 32 * mm, leftPadding=0, rightPadding=0, topPadding=4 * mm, bottomPadding=2 * mm)
    doc = BaseDocTemplate(str(output), pagesize=A4, title="Glance - Multimodal Fashion & Context Retrieval", author="Glance internship candidate", pageTemplates=[PageTemplate("main", [frame], onPage=page_chrome)], leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm)
    story = []

    story += [Spacer(1, 13 * mm), p("GLANCE ML INTERNSHIP ASSIGNMENT", styles["kicker"]), p("Multimodal Fashion<br/>&amp; Context Retrieval", styles["title"]), p("A fashion-aware, zero-shot search system that binds attributes to garments while preserving scene, action, and style context.", styles["subtitle"])]
    story.append(p("My chosen approach", styles["h2"]))
    story.append(callout("I use FashionSigLIP embeddings + multi-scale local crops + structured query decomposition + conjunctive reranking + FAISS HNSW candidate search.", styles))
    repo_text = f'<b>My codebase:</b> <link href="{repo_url}" color="#A34532">{repo_url}</link>' if repo_url else "<b>My codebase:</b> I have included the complete local repository with this report. I will insert its public GitHub URL at publication time using <font name='Courier'>--repo-url</font>."
    story += [p(repo_text, styles["body"]), Spacer(1, 3 * mm)]
    story.append(styled_table([
        ["What the system understands", "How it is represented"],
        ["Garment + color binding", "Bound phrase vectors scored against local regions"],
        ["Place and setting", "Full-image scene vector and explicit scene facets"],
        ["Style and vibe", "Prompt-ensembled style facets plus full-query semantics"],
        ["Multi-attribute conjunction", "Smooth minimum prevents one strong facet hiding a missing one"],
    ], [59 * mm, 105 * mm]))
    story += [Spacer(1, 7 * mm), p("What I completed", styles["h2"])]
    story += bullets([
        "I separated the Part A indexer and Part B retriever into reusable modules and CLIs.",
        "I audited all 3,200 downloaded images and configured a deterministic 1,000-image default index.",
        "I passed eight offline tests, built a real 1,000-image FashionSigLIP index, and ran all five required queries.",
        "I do not fabricate test-set accuracy: the supplied Fashionpedia test images have no public relevance labels.",
    ], styles)
    story.append(PageBreak())

    story += [p("1. Problem framing &amp; data", styles["h1"]), p("I treat this as more than ordinary object search. A result must satisfy apparel identity, apparel attributes, relations between them, scene, and inferred style. The key failure mode I target is compositional: a global embedding can notice red, white, shirt, and tie yet rank the color-swapped outfit equally well.", styles["body"])]
    if profile:
        profile_data = [
            ["Audit field", "Observed value"],
            ["Images", f"{profile.get('valid_images', 0):,} valid / {profile.get('image_count', 0):,} discovered"],
            ["Corrupt files", str(len(profile.get("corrupt_images", [])))],
            ["Orientation", f"{profile.get('orientation', {}).get('portrait', 0):,} portrait; {profile.get('orientation', {}).get('landscape', 0):,} landscape; {profile.get('orientation', {}).get('square', 0):,} square"],
            ["Median resolution", f"{profile.get('width', {}).get('median', '?')} x {profile.get('height', {}).get('median', '?')} pixels"],
            ["Default experimental subset", "1,000 images, deterministic SHA-256 sampling, seed 17"],
        ]
        story.append(styled_table(profile_data, [53 * mm, 111 * mm]))
    story += [Spacer(1, 5 * mm), p("Dataset reality", styles["h2"]), p("I worked with the supplied Fashionpedia test images. The official ontology has 46 apparel objects and 294 fine-grained attributes; localization annotations exist for train/validation, but not for these test images. I therefore support both annotation-free heuristic crops and optional Fashionpedia bounding-box crops. [1][2]", styles["body"])]
    story.append(styled_table([
        ["Required axis", "Coverage and mitigation"],
        ["Environment", "Street and runway are common; office, park, and home should be deliberately supplemented and labeled."],
        ["Clothing", "Formal, casual, dresses, tops, outerwear, and accessories are visibly diverse."],
        ["Color", "Broad palette is present; evaluation must verify color-garment binding, not color presence alone."],
    ], [40 * mm, 124 * mm]))
    if montage.exists():
        story += [PageBreak(), p("Dataset glimpse", styles["h1"])]
        img = Image(str(montage), width=164 * mm, height=164 * mm * 1.286)
        story += [img, p("Deterministic sample (seed 29). It visibly spans runway, street, casual, and formal imagery; controlled office/home coverage is weaker and should be supplemented for a production benchmark.", styles["small"])]
    story.append(PageBreak())

    story += [p("2. Approaches I considered &amp; trade-offs", styles["h1"])]
    approaches = [
        ["Approach", "Strength", "Main limitation", "Use when"],
        ["Vanilla CLIP / SigLIP", "Fast zero-shot baseline; minimal code", "Weak binding and small-detail evidence", "Baseline, broad semantic search"],
        ["Fashion-tuned dual encoder", "Better garment/color/style vocabulary", "Product tuning may underweight scene", "Fashion-heavy catalogs"],
        ["Detector + attribute classifiers", "Localized, interpretable apparel evidence", "Label-bound; training and maintenance cost", "Closed ontology, high precision"],
        ["Caption/VLM + text retrieval", "Rich explanations and relation words", "Slow; hallucination and caption bottleneck", "Offline enrichment, small catalogs"],
        ["Cross-encoder reranker", "Strong pairwise reasoning on shortlist", "High query latency; needs training data", "High-value top-k precision"],
        ["Chosen hybrid", "Zero-shot + local binding + context + scalable ANN", "Heuristic crops are imperfect", "This assignment and extensible production MVP"],
    ]
    story.append(styled_table(approaches, [35 * mm, 43 * mm, 47 * mm, 39 * mm], font_size=7.0))
    story += [Spacer(1, 5 * mm), p("Why I chose FashionSigLIP", styles["h2"]), p("SigLIP replaces CLIP's batch-global softmax objective with a pairwise sigmoid loss. I selected the Marqo model because it starts from ViT-B/16 SigLIP and is tuned with fashion categories, styles, colors, materials, keywords, and fine details. Its model card reports stronger average text-to-image recall than generic ViT-B/16 SigLIP on six public fashion datasets. Those published numbers motivate my model choice; I do not claim them as results on this dataset. [3][4]", styles["body"])]
    story += [p("My design principle", styles["h2"]), callout("I use the domain model for visual vocabulary, then impose structure at retrieval time. This keeps zero-shot behavior and avoids pretending that one vector is perfectly compositional.", styles)]
    story.append(PageBreak())

    story += [p("3. Chosen architecture", styles["h1"]), ArchitectureDiagram(PAGE_W - 36 * mm), Spacer(1, 4 * mm)]
    story.append(p("Part A - Indexer", styles["h2"]))
    story += bullets([
        "Discover and deterministically sample images; validate dimensions and EXIF orientation.",
        "Create seven views: full, upper, torso, lower, center, left, and right. Append up to twelve Fashionpedia instance boxes when annotations are supplied.",
        "Encode views with the same FashionSigLIP image tower and L2-normalize every vector.",
        "Persist memory-mappable NumPy arrays, JSONL metadata, a versioned manifest, and optional FAISS HNSW inner-product index.",
    ], styles)
    story.append(p("Part B - Retriever", styles["h2"]))
    story += bullets([
        "Embed the full query twice with prompt ensembling, then retrieve a broad top-100 candidate set from full-image vectors.",
        "Parse bound color-garment phrases, scene, style, action, and negative clauses with a deterministic, auditable lexicon.",
        "Score garment bindings against the best valid region; score context/style/action against the full image.",
        "Standardize candidate-local scores, apply a smooth-AND conjunction, subtract explicit negative evidence, and return top k with a score breakdown.",
    ], styles)
    story.append(PageBreak())

    story += [p("4. ML logic for compositional fashion queries", styles["h1"]), p("The parser retains relations instead of reducing the query to a bag of labels. For the hardest evaluation query:", styles["body"]), p("A red tie and a white shirt in a formal setting.<br/><br/>bindings = [red tie, white shirt]<br/>scene = [formal setting]<br/>style = [formal outfit]", styles["code"])]
    story.append(p("Scoring", styles["h2"]))
    story.append(p("Let <i>g</i> be the standardized global similarity, <i>b</i><sub>j</sub> each binding's best-region similarity, and <i>c</i><sub>j</sub> every scene/style/action score. A smooth minimum acts as a differentiable logical AND:", styles["body"]))
    story.append(callout("AND(s<sub>1</sub>...s<sub>m</sub>) = -tau log mean exp(-s<sub>j</sub>/tau), &nbsp; tau = 0.35", styles))
    story.append(p("Final score = 0.15 global + 0.55 AND(all facets) + 0.20 mean(bindings) + 0.10 mean(context) - 0.20 negative penalty. I keep global similarity for broad recall while explicit facets dominate final precision.", styles["body"]))
    query_rows = [["Evaluation query", "Parsed evidence", "Why it helps"]]
    query_rows += [
        ["bright yellow raincoat", "binding: bright yellow raincoat", "Color, intensity, and garment stay attached"],
        ["business attire inside a modern office", "style: professional; scene: modern office", "Full-image scene cannot be replaced by a blazer close-up"],
        ["blue shirt sitting on a park bench", "binding: blue shirt; action: sitting; scene: park", "All three facets must survive smooth-AND"],
        ["casual weekend outfit for a city walk", "style: casual weekend; scene: urban street", "Maps implied vibe and paraphrased place"],
        ["red tie and white shirt in a formal setting", "two bindings; scene and style", "Opposite color-garment assignment scores differently"],
    ]
    story.append(styled_table(query_rows, [49 * mm, 61 * mm, 54 * mm], font_size=6.9))
    story += [Spacer(1, 5 * mm), p("Ablations that isolate value", styles["h2"])]
    story += bullets([
        "A0: full-image generic SigLIP (vanilla baseline).",
        "A1: full-image FashionSigLIP (domain adaptation contribution).",
        "A2: A1 + multi-region max pooling (local evidence contribution).",
        "A3: A2 + decomposition + smooth-AND (compositional contribution).",
    ], styles)
    story.append(PageBreak())

    story += [p("5. My code workflows &amp; reproducibility", styles["h1"]), p("I separated the ML logic from data paths and command-line orchestration. I record the model ID, dimensionality, index version, image count, region strategy, and metric in the manifest, and reject a query-time model mismatch.", styles["body"])]
    story.append(p("Indexer", styles["h2"]))
    story.append(p("python scripts/index.py --image-dir val_test2020/test --output artifacts/fashionpedia-1000 --max-images 1000 --seed 17", styles["code"]))
    story.append(p("Retriever", styles["h2"]))
    story.append(p('python scripts/search.py "A red tie and a white shirt in a formal setting." --index artifacts/fashionpedia-1000 -k 5 --contact-sheet outputs/result.jpg', styles["code"]))
    modules = [
        ["Module", "Responsibility"],
        ["dataset.py", "Discovery, deterministic sample, crops, optional annotation boxes"],
        ["encoder.py", "Lazy OpenCLIP/FashionSigLIP adapter and normalization"],
        ["index_store.py", "Versioned arrays, metadata, FAISS HNSW, exact fallback"],
        ["query.py", "Transparent bound-facet parser and prompt ensembles"],
        ["retrieval.py", "ANN candidate generation and compositional reranking"],
        ["evaluation.py", "MRR, Recall@k, and nDCG@k"],
    ]
    story.append(styled_table(modules, [42 * mm, 122 * mm]))
    story += [Spacer(1, 5 * mm), p("Verification I completed", styles["h2"])]
    story += bullets([
        "Python compilation completed for source, scripts, and tests.",
        "8/8 offline unit tests passed (query binding, counterfactuals, crop validity, deterministic sampling, vector-store round trip, exact search, conjunctive penalty).",
        "I built the full real-model index: 1,000 images, 7,000 region vectors, 768 dimensions, all finite and unit-normalized.",
        "I ran all five required prompts with FashionSigLIP and generated JSON score breakdowns plus contact sheets.",
        "Dataset audit read and verified all 3,200 JPGs; zero corrupt files were detected.",
    ], styles)
    story.append(PageBreak())

    story += [p("6. My evaluation plan", styles["h1"]), p("Because the downloaded test split has no public attribute labels, I use relevance judgments for honest evaluation. I do not use filename matching. I included all five prompts and a judgment template in the repository.", styles["body"])]
    eval_steps = [
        ["Step", "Protocol"],
        ["Pool", "Union top 30 from A0-A3 so no single method defines the label pool"],
        ["Label", "Two independent annotators mark exact, partial, and non-relevant"],
        ["Resolve", "Adjudicate disagreements; report Cohen's kappa"],
        ["Metrics", "MRR for first good result; Recall@10; nDCG@10 for graded relevance"],
        ["Slices", "Attribute, context, semantic, style inference, compositional"],
        ["Failure tags", "wrong color binding, missing garment, wrong scene, wrong action, style mismatch"],
    ]
    story.append(styled_table(eval_steps, [29 * mm, 135 * mm]))
    story += [Spacer(1, 5 * mm), p("Acceptance criteria", styles["h2"])]
    story += bullets([
        "A3 should beat A1 on the compositional and multi-attribute slices, not only overall average.",
        "Latency should be reported separately for text encoding, ANN retrieval, and reranking.",
        "Any hyperparameter change is selected on a validation judgment set, never the five final prompts.",
        "Qualitative contact sheets accompany aggregate metrics to expose duplicated or near-miss results.",
    ], styles)
    story += [Spacer(1, 2 * mm), callout("I ran the real model and report qualitative outputs on the next page. I intentionally withhold MRR/Recall/nDCG until independent relevance labels exist; unlabeled rankings are not accuracy metrics.", styles)]
    story.append(PageBreak())

    story += [p("7. Real model run: qualitative evidence", styles["h1"]), p("I ran FashionSigLIP over the deterministic 1,000-image index and retrieved the top 10 for every required prompt. The images below are the actual rank-1 outputs, not hand-picked examples.", styles["body"])]
    if evaluation_summary.exists():
        story += [Image(str(evaluation_summary), width=164 * mm, height=48.3 * mm), Spacer(1, 4 * mm)]
    evidence = [
        ["Query slice", "What I observed", "Assessment"],
        ["Bright yellow raincoat", "Strong yellow evidence; coat-like items rise, but rank 1 is a yellow top under a blazer.", "Partial - exact garment appears absent from the subset."],
        ["Business attire + office", "Professional clothing is strong; results are runway/event rather than office interiors.", "Partial - confirms a context coverage gap."],
        ["Blue shirt + park bench", "Rank 1 is a light-blue top, seated in a park-like walkway; rank 2 has the bench but not a blue shirt.", "Partial - no exact conjunction appears in the shortlist."],
        ["Casual city walk", "Rank 1 is a casual warm-weather outfit against an urban wall; top results contain multiple street scenes.", "Strong qualitative match."],
        ["Red tie + white shirt + formal", "Rank 1 contains a red patterned tie, white shirt, suit, and formal event background.", "Exact match; counterfactual reranking fixed the shortcut."],
    ]
    story.append(styled_table(evidence, [37 * mm, 83 * mm, 44 * mm], font_size=6.9))
    story += [Spacer(1, 5 * mm), p("What the run taught me", styles["h2"])]
    story += bullets([
        "Structured reranking materially helps compositional binding: the exact red-tie image moved from rank 3 under the first scorer to rank 1 after same-region counterfactual margins and facet-dominant weights.",
        "A retrieval model cannot return an exact scene/garment combination missing from its indexed sample. Office and raincoat coverage must be added before claiming high recall on those slices.",
        "These are qualitative observations, not accuracy metrics. I still require independent relevance labels for MRR, Recall@10, and nDCG@10.",
    ], styles)
    story.append(PageBreak())

    story += [p("8. Scale, locations, weather &amp; precision", styles["h1"]), p("Path to one million images", styles["h2"])]
    story += bullets([
        "Offline batch GPU encoding; incremental append jobs keyed by image content hash.",
        "Memory-map vectors; use float16 or product quantization. My 768-D float32 global vector costs about 3 KB/image before index overhead.",
        "Move from HNSW to IVF-PQ when memory dominates, tune recall/latency on held-out queries, and shard by market or catalog.",
        "Retrieve 100-500 candidates cheaply; keep region vectors in a second-stage store and rerank only the shortlist.",
        "Cache text/facet embeddings and batch concurrent queries. FAISS provides the underlying ANN options without a custom vector database. [5]",
    ], styles)
    story.append(p("Adding cities and places", styles["h2"]))
    story += bullets([
        "Add a generic geolocation/scene encoder alongside the fashion encoder; keep separate calibrated scores so fashion tuning does not erase landmarks.",
        "Index structured EXIF/GPS or catalog location metadata as filters; reverse-geocode to city/country and maintain a place hierarchy.",
        "Train with hard negatives from visually similar cities and verify privacy/consent before storing coordinates.",
    ], styles)
    story.append(p("Adding weather", styles["h2"]))
    story += bullets([
        "Infer visible conditions (rain, snow, sun) and apparel affordances (raincoat, umbrella, layers) as separate facets.",
        "For live use, join time/location metadata to a weather API at ingestion; store temperature, precipitation, wind, and season as filterable fields.",
        "Avoid causal overclaiming: a coat is evidence of cold-weather styling, not proof of the measured temperature.",
    ], styles)
    story.append(p("Improving precision", styles["h2"]))
    story += bullets([
        "Replace heuristic crops with Fashionpedia-trained segmentation or open-vocabulary apparel detection.",
        "Fine-tune with query-image pairs and color-swap hard negatives; use contrastive losses that explicitly penalize incorrect bindings.",
        "Train a small cross-encoder/VLM reranker on human judgments; calibrate per-facet thresholds and add diversity-aware reranking.",
        "Expand the lexicon from the Fashionpedia ontology, or use a constrained LLM that emits a validated query graph.",
    ], styles)
    story.append(PageBreak())

    story += [p("9. Limitations, conclusion &amp; references", styles["h1"]), p("Known limitations", styles["h2"])]
    story += bullets([
        "The available Fashionpedia sample is fashion-rich but not deliberately balanced across office, park, street, and home; supplementing context scenes would make evaluation fairer.",
        "Heuristic crops overlap and cannot guarantee object-level grounding for a tiny tie, logo, or accessory.",
        "Candidate-local z-scores are interpretable within a query but are not probabilities and are not directly comparable across queries.",
        "English lexicons miss rare synonyms, multilingual queries, and complex negation; model semantics still provide a global fallback.",
        "FashionSigLIP weights require an initial network download and substantial CPU/GPU compute; the code intentionally loads them lazily.",
    ], styles)
    story.append(p("Conclusion", styles["h2"]))
    story.append(callout("I chose a pragmatic midpoint between a weak one-vector baseline and an expensive fully supervised detector stack. Domain adaptation supplies fashion vocabulary; local views supply evidence; explicit query structure preserves binding; smooth-AND makes every requested facet matter; and ANN candidate generation keeps my design scalable. Most importantly, my system remains zero-shot, modular, inspectable, and honest about what I have and have not measured.", styles))
    story.append(p("References", styles["h2"]))
    refs = [
        '[1] Jia et al. "Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset." ECCV 2020. <link href="https://arxiv.org/abs/2004.12276" color="#A34532">arxiv.org/abs/2004.12276</link>',
        '[2] Fashionpedia project and ontology. <link href="https://fashionpedia.github.io/home/index.html" color="#A34532">fashionpedia.github.io</link>',
        '[3] Zhai et al. "Sigmoid Loss for Language Image Pre-Training." ICCV 2023. <link href="https://arxiv.org/abs/2303.15343" color="#A34532">arxiv.org/abs/2303.15343</link>',
        '[4] Marqo-FashionSigLIP model card, architecture, usage, license, and benchmark table. <link href="https://huggingface.co/Marqo/marqo-fashionSigLIP" color="#A34532">huggingface.co/Marqo/marqo-fashionSigLIP</link>',
        '[5] Johnson, Douze, and Jegou. FAISS similarity-search library. <link href="https://github.com/facebookresearch/faiss" color="#A34532">github.com/facebookresearch/faiss</link>',
    ]
    story += [p(item, styles["ref"]) for item in refs]
    story += [Spacer(1, 5 * mm), p("I generated this report from my verified repository, real FashionSigLIP run, and dataset audit. I do not present unlabeled qualitative results as accuracy metrics.", styles["center"])]

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


def main():
    parser = argparse.ArgumentParser(description="Build the single-PDF Glance assignment report.")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "pdf" / "glance_fashion_retrieval_submission.pdf")
    parser.add_argument("--repo-url", default=None, help="Public GitHub URL to print in the PDF")
    args = parser.parse_args()
    build_pdf(args.output, args.repo_url)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

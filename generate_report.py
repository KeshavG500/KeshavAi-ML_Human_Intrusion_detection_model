"""
generate_report.py
──────────────────
Generates a professional Word (.docx) documentation report for the
Human Intrusion Detection System model.
"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import os

# ── Paths ─────────────────────────────────────────────────────────────────
TRAIN_DIR = r"runs\train\intrusion_yolov8s_person"
OUTPUT_PATH = r"Human_Intrusion_Detection_Model_Report.docx"

IMG = {
    "results":        os.path.join(TRAIN_DIR, "results.png"),
    "pr_curve":       os.path.join(TRAIN_DIR, "BoxPR_curve.png"),
    "f1_curve":       os.path.join(TRAIN_DIR, "BoxF1_curve.png"),
    "conf_matrix":    os.path.join(TRAIN_DIR, "confusion_matrix.png"),
    "conf_matrix_n":  os.path.join(TRAIN_DIR, "confusion_matrix_normalized.png"),
    "labels":         os.path.join(TRAIN_DIR, "labels.jpg"),
    "val0_labels":    os.path.join(TRAIN_DIR, "val_batch0_labels.jpg"),
    "val0_pred":      os.path.join(TRAIN_DIR, "val_batch0_pred.jpg"),
    "val1_pred":      os.path.join(TRAIN_DIR, "val_batch1_pred.jpg"),
    "train_last":     os.path.join(TRAIN_DIR, "train_batch17850.jpg"),
}


# ── Helpers ───────────────────────────────────────────────────────────────

def set_cell_shading(cell, color_hex):
    """Set background color of a table cell."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def add_styled_table(doc, headers, rows, col_widths=None, header_color="1B2A4A"):
    """Add a formatted table with colored header row."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(cell, header_color)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(9)
            # Alternate row shading
            if r_idx % 2 == 0:
                set_cell_shading(cell, "F0F4FA")

    # Set column widths
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Cm(w)

    doc.add_paragraph()  # spacer
    return table


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1B, 0x2A, 0x4A)
    return h


def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.style.font.size = Pt(10)
    return p


def add_image_safe(doc, path, width=Inches(5.8), caption=None):
    """Add an image if the file exists, with an optional caption."""
    if not os.path.exists(path):
        doc.add_paragraph(f"[Image not found: {path}]")
        return
    doc.add_picture(path, width=width)
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True if cap.runs else None
        for run in cap.runs:
            run.font.size = Pt(8)
            run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
            run.italic = True


def add_note_box(doc, text, prefix="ℹ️  NOTE"):
    """Add a styled note/callout paragraph."""
    p = doc.add_paragraph()
    run = p.add_run(f"{prefix}: ")
    run.bold = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x1B, 0x2A, 0x4A)
    run2 = p.add_run(text)
    run2.font.size = Pt(9)
    run2.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


# ── Build Document ────────────────────────────────────────────────────────

def build_report():
    doc = Document()

    # ── Page margins
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # ── Default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)
    style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # ──────────────────────── TITLE PAGE ──────────────────────────────────

    for _ in range(6):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("🛡️ Human Intrusion Detection System")
    run.font.size = Pt(26)
    run.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x2A, 0x4A)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Model Documentation Report")
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0x4A, 0x6C, 0xF7)

    doc.add_paragraph()

    meta_items = [
        ("Version", "1.0"),
        ("Date", "June 2026"),
        ("Hardware", "NVIDIA Quadro T2000 (4GB VRAM)"),
        ("Framework", "Python 3.13 · PyTorch 2.7.1 · CUDA 11.8"),
        ("Model", "YOLOv8s + ByteTrack"),
    ]
    for label, value in meta_items:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p.add_run(f"{label}: ")
        r1.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        r2 = p.add_run(value)
        r2.font.size = Pt(11)
        r2.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    doc.add_page_break()

    # ──────────────────────── TABLE OF CONTENTS (placeholder) ─────────────

    add_heading(doc, "Table of Contents", level=1)
    toc_items = [
        "1.  Executive Summary",
        "2.  Core Model Algorithm",
        "3.  Datasets Used",
        "4.  Preprocessing Pipeline",
        "5.  Data Augmentation",
        "6.  Training Configuration",
        "7.  Evaluation Metrics & Scores",
        "8.  Training Curves & Graphs",
        "9.  Confusion Matrix",
        "10. Sample Test Predictions",
        "11. End-to-End Inference Pipeline",
        "12. Edge Deployment Strategy",
        "13. Limitations & Future Improvements",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)
        for run in p.runs:
            run.font.size = Pt(11)

    doc.add_page_break()

    # ──────────────────────── 1. EXECUTIVE SUMMARY ────────────────────────

    add_heading(doc, "1. Executive Summary", level=1)
    add_body(doc,
        "This document describes the Human Intrusion Detection System, a real-time computer vision "
        "pipeline designed to detect unauthorized persons in restricted zones using CCTV/IP camera feeds. "
        "The system combines a YOLOv8s object detector with ByteTrack multi-object tracking, "
        "polygon-based zone logic, and an alert engine suitable for edge deployment."
    )

    add_styled_table(doc,
        ["Component", "Technology"],
        [
            ["Object Detection",      "YOLOv8s (Ultralytics)"],
            ["Multi-Object Tracking",  "ByteTrack (IoU-based)"],
            ["Zone Enforcement",       "Shapely polygon geometry"],
            ["Edge Runtime",           "ONNX Runtime + CUDA"],
            ["Target Platform",        "NVIDIA Jetson / x86 GPU edge boxes"],
        ],
        col_widths=[5, 10],
    )

    # ──────────────────────── 2. CORE MODEL ALGORITHM ─────────────────────

    add_heading(doc, "2. Core Model Algorithm", level=1)

    add_heading(doc, "2.1 YOLOv8s — Object Detection Backbone", level=2)
    add_body(doc,
        "The detection backbone is YOLOv8s (small variant) from Ultralytics, a single-stage "
        "anchor-free object detector. It uses a CSPDarknet53 backbone, FPN+PAN neck, and a "
        "decoupled detection head with separate classification and regression branches."
    )

    add_styled_table(doc,
        ["Property", "Value"],
        [
            ["Model Variant",      "YOLOv8s (small)"],
            ["Total Parameters",   "11,125,971"],
            ["GFLOPs",             "28.4"],
            ["Fused Layers",       "73"],
            ["Detection Type",     "Anchor-free"],
            ["Number of Classes",  "1 (person only)"],
            ["Input Resolution",   "640 × 640 pixels"],
        ],
        col_widths=[5, 10],
    )

    add_heading(doc, "Internal Flow", level=3)
    add_body(doc,
        "1. Backbone (CSPDarknet53): Extracts hierarchical features at 3 scales (P3/8, P4/16, P5/32) "
        "using Cross-Stage Partial connections and C2f modules.\n"
        "2. Neck (FPN + PAN): Feature Pyramid Network (top-down) + Path Aggregation Network (bottom-up) "
        "fuse multi-scale features for detecting both small and large persons.\n"
        "3. Decoupled Head: Separate classification and regression branches. The classification branch "
        "predicts class probabilities, the regression branch predicts bounding box coordinates using "
        "Distribution Focal Loss (DFL).\n"
        "4. Post-Processing: Non-Maximum Suppression (NMS) at IoU threshold 0.45 filters overlapping detections."
    )

    add_heading(doc, "2.2 ByteTrack — Multi-Object Tracking", level=2)
    add_body(doc,
        "After detection, ByteTrack assigns persistent track IDs to each detected person across "
        "consecutive frames using a two-stage association algorithm."
    )

    add_styled_table(doc,
        ["Stage", "Input", "Match Threshold", "Purpose"],
        [
            ["Stage 1", "High-confidence detections (≥ 0.6)", "IoU ≥ 0.8",  "Match strong detections to confirmed tracks"],
            ["Stage 2", "Low-confidence detections (0.5–0.6)", "IoU ≥ 0.56", "Recover partially occluded / fading tracks"],
        ],
        col_widths=[2.5, 5, 3, 5],
    )

    add_heading(doc, "Track Lifecycle", level=3)
    add_body(doc,
        "• TENTATIVE → seen but not yet confirmed (< 3 hits)\n"
        "• CONFIRMED → matched for ≥ 3 consecutive frames\n"
        "• LOST → not matched for > 30 frames → deleted"
    )

    add_heading(doc, "2.3 Zone Enforcement (ZoneManager)", level=2)
    add_body(doc,
        "The ZoneManager uses Shapely polygon geometry to define restricted areas in the camera frame:\n"
        "1. Zone Entry: Checks if a person's centroid falls inside a polygon using Point.within(Polygon).\n"
        "2. Boundary Crossing: Detects when a tracked person's trajectory crosses a boundary LineString.\n"
        "3. Time-Restricted Enforcement: Zones can have active hours (e.g., 22:00–06:00).\n"
        "4. Alert Cooldown: Prevents alert flooding with a configurable cooldown (default 10 seconds)."
    )

    doc.add_page_break()

    # ──────────────────────── 3. DATASETS USED ────────────────────────────

    add_heading(doc, "3. Datasets Used", level=1)
    add_body(doc,
        "The model was trained on person-only annotations extracted from publicly available datasets."
    )

    add_heading(doc, "3.1 MS COCO 2017 (Primary Training Dataset)", level=2)
    add_styled_table(doc,
        ["Property", "Details"],
        [
            ["Description",    "Large-scale object detection benchmark with 80 classes"],
            ["Used Class",     "person (class ID 0) only"],
            ["Train Images",   "http://images.cocodataset.org/zips/train2017.zip (~18 GB, 118K images)"],
            ["Val Images",     "http://images.cocodataset.org/zips/val2017.zip (~1 GB, 5K images)"],
            ["Annotations",    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"],
            ["License",        "Creative Commons Attribution 4.0"],
        ],
        col_widths=[4, 11],
    )

    add_heading(doc, "3.2 MOT17 (Tracking Benchmark)", level=2)
    add_styled_table(doc,
        ["Property", "Details"],
        [
            ["Description", "Multi-Object Tracking benchmark with pedestrian annotations"],
            ["Download",    "https://motchallenge.net/data/MOT17.zip"],
            ["Usage",       "Supplementary person detection data with tracking annotations"],
        ],
        col_widths=[4, 11],
    )

    add_heading(doc, "3.3 VIRAT (Surveillance Video Dataset)", level=2)
    add_styled_table(doc,
        ["Property", "Details"],
        [
            ["Description", "Real-world surveillance video dataset from DARPA"],
            ["Info Page",   "https://viratdata.org/"],
            ["Usage",       "Reference for surveillance-specific person pose, scale, and occlusion"],
        ],
        col_widths=[4, 11],
    )

    add_heading(doc, "3.4 Actual Training Split", level=2)
    add_styled_table(doc,
        ["Split", "Images", "Purpose"],
        [
            ["Train",      "~70%", "Model weight optimization"],
            ["Validation", "~15%", "Hyperparameter tuning + early stopping"],
            ["Test",       "~15%", "Final held-out evaluation"],
        ],
        col_widths=[4, 4, 7],
    )
    add_body(doc, "Total images used for training: 2,546 images with 7,958 person instances.")

    add_heading(doc, "3.5 Label Distribution", level=2)
    add_image_safe(doc, IMG["labels"], Inches(5.5),
        "Figure 1: Label distribution showing bounding box positions, sizes, and aspect ratios.")

    doc.add_page_break()

    # ──────────────────────── 4. PREPROCESSING ────────────────────────────

    add_heading(doc, "4. Preprocessing Pipeline", level=1)

    add_heading(doc, "4.1 Image Preprocessing", level=2)
    add_styled_table(doc,
        ["Step", "Technique", "Details"],
        [
            ["Resize",        "Letterbox Scaling",   "LongestMaxSize(640) + pad to 640×640 with fill (114,114,114)"],
            ["Normalization", "ImageNet Standard",   "Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]"],
            ["Color Space",   "BGR → RGB",           "OpenCV BGR input converted to RGB for model input"],
            ["Tensor",        "ToTensorV2()",        "HWC uint8 → CHW float32 tensor"],
        ],
        col_widths=[3, 4, 8],
    )

    add_heading(doc, "4.2 Real-Time Frame Extraction", level=2)
    add_body(doc,
        "The FrameExtractor module provides a multi-threaded buffered capture system:\n"
        "• Input Sources: RTSP, RTMP, USB webcam, HTTP MJPEG, video files\n"
        "• Target FPS: 15 fps (configurable)\n"
        "• Resolution: 1280×720 → resized to 640×640 for inference\n"
        "• Buffer Strategy: Bounded queue (size=4). When full, oldest frame is dropped "
        "to always process the latest frame (critical for real-time latency)."
    )

    # ──────────────────────── 5. DATA AUGMENTATION ────────────────────────

    add_heading(doc, "5. Data Augmentation", level=1)

    add_heading(doc, "5.1 YOLOv8 Built-in Augmentations", level=2)
    add_styled_table(doc,
        ["Augmentation", "Parameter", "Value", "Purpose"],
        [
            ["HSV Hue",         "hsv_h",       "0.015", "Simulate lighting changes"],
            ["HSV Saturation",  "hsv_s",       "0.7",   "Handle color variation"],
            ["HSV Value",       "hsv_v",       "0.4",   "Brightness robustness"],
            ["Scale",           "scale",       "0.5",   "Multi-scale person sizes"],
            ["Translate",       "translate",   "0.1",   "Positional invariance"],
            ["Horizontal Flip", "fliplr",      "0.5",   "Left-right symmetry"],
            ["Mosaic",          "mosaic",      "1.0",   "4-image mosaic (100% probability)"],
            ["MixUp",           "mixup",       "0.1",   "Blended image regularization"],
            ["Copy-Paste",      "copy_paste",  "0.2",   "Instance-level augmentation"],
            ["Erasing",         "erasing",     "0.4",   "Random erasing (occlusion sim.)"],
            ["Auto Augment",    "auto_augment","RandAugment", "Automated augmentation policy"],
        ],
        col_widths=[3.5, 3, 2.5, 6],
    )

    add_heading(doc, "5.2 Albumentations Custom Pipeline", level=2)
    add_styled_table(doc,
        ["Category", "Techniques", "Probability"],
        [
            ["Spatial",      "ShiftScaleRotate (±10°), Perspective",                           "0.7 / 0.3"],
            ["Photometric",  "RandomBrightnessContrast, HueSaturationValue, CLAHE, RGBShift",  "0.8"],
            ["Noise/Blur",   "GaussNoise, ISONoise, MotionBlur, GaussianBlur, MedianBlur",     "0.3–0.4"],
            ["Dropout",      "CoarseDropout (8 holes, 64px)",                                  "0.3"],
            ["Weather",      "RandomRain, RandomFog, RandomShadow",                            "0.2"],
        ],
        col_widths=[3, 8, 3],
    )
    add_note_box(doc,
        "The weather simulation augmentations (rain, fog, shadow) are specifically included "
        "to make the model robust for outdoor surveillance scenarios.",
        "💡  TIP"
    )

    doc.add_page_break()

    # ──────────────────────── 6. TRAINING CONFIGURATION ───────────────────

    add_heading(doc, "6. Training Configuration", level=1)

    add_heading(doc, "6.1 Hyperparameters", level=2)
    add_styled_table(doc,
        ["Parameter", "Value", "Rationale"],
        [
            ["Base Model",     "YOLOv8s (pre-trained on COCO)",     "Transfer learning from COCO weights"],
            ["Epochs",         "80",                                 "Sufficient convergence with early stopping"],
            ["Batch Size",     "8",                                  "Maximum for 4GB VRAM with 640px images"],
            ["Optimizer",      "AdamW",                              "Better generalization than SGD"],
            ["Initial LR",    "0.001",                               "Standard for fine-tuning"],
            ["Final LR",      "0.01 (of initial)",                   "Cosine-like decay to 0.00001"],
            ["Momentum",       "0.937",                              "Smooth gradient updates"],
            ["Weight Decay",   "0.0005",                             "L2 regularization"],
            ["Warmup Epochs",  "3",                                  "Gradual LR ramp-up"],
            ["Patience",       "20",                                 "Early stopping on val mAP plateau"],
            ["Image Size",     "640 × 640",                          "Balance between accuracy and speed"],
            ["AMP",            "Enabled",                            "Automatic Mixed Precision for VRAM savings"],
            ["Workers",        "4",                                  "DataLoader parallel workers"],
            ["Seed",           "0",                                  "Reproducibility"],
        ],
        col_widths=[3.5, 5, 6.5],
    )

    add_heading(doc, "6.2 Loss Functions", level=2)
    add_styled_table(doc,
        ["Loss", "Weight", "Purpose"],
        [
            ["Box Loss (CIoU)",             "7.5", "Bounding box regression quality"],
            ["Classification Loss (BCE)",   "0.5", "Person/background classification"],
            ["DFL Loss (Distribution Focal)","1.5", "Precise box coordinate distributions"],
        ],
        col_widths=[5, 3, 7],
    )

    add_heading(doc, "6.3 Post-Processing", level=2)
    add_styled_table(doc,
        ["Parameter", "Value"],
        [
            ["Confidence Threshold",         "0.50"],
            ["IoU Threshold (NMS)",          "0.45"],
            ["Max Detections per Image",     "300 (train) / 50 (deploy)"],
        ],
        col_widths=[6, 9],
    )

    doc.add_page_break()

    # ──────────────────────── 7. EVALUATION METRICS ───────────────────────

    add_heading(doc, "7. Evaluation Metrics & Scores", level=1)

    add_heading(doc, "7.1 Final Model Performance (Best Weights)", level=2)
    add_styled_table(doc,
        ["Metric", "Score", "Meaning"],
        [
            ["mAP@0.5",        "94.4%",  "Mean Average Precision at IoU ≥ 0.50 — primary detection quality metric"],
            ["mAP@0.5:0.95",   "77.6%",  "Stricter mAP averaged over IoU thresholds 0.50 to 0.95 (COCO standard)"],
            ["Precision",      "92.4%",  "Of all detections the model made, 92.4% were actual persons"],
            ["Recall",         "88.2%",  "Of all actual persons in images, 88.2% were correctly detected"],
            ["F1-Score",       "~90.3%", "Harmonic mean of Precision and Recall"],
        ],
        col_widths=[3.5, 2, 9.5],
    )

    add_heading(doc, "7.2 Metric Explanations", level=2)

    add_heading(doc, "Precision (92.4%)", level=3)
    add_body(doc,
        '"How many of the model\'s detections are correct?"\n\n'
        "Precision = True Positives / (True Positives + False Positives)\n\n"
        "A precision of 92.4% means that for every 100 bounding boxes the model draws, ~92 are real "
        "persons and only ~8 are false alarms. High precision = fewer false alarms in the deployed system."
    )

    add_heading(doc, "Recall (88.2%)", level=3)
    add_body(doc,
        '"How many actual persons does the model find?"\n\n'
        "Recall = True Positives / (True Positives + False Negatives)\n\n"
        "A recall of 88.2% means the model catches ~88 out of every 100 real persons. "
        "High recall = fewer missed intruders. The remaining ~12% are typically small/occluded persons."
    )

    add_heading(doc, "mAP@0.5 (94.4%)", level=3)
    add_body(doc,
        '"Overall detection quality at standard IoU threshold"\n\n'
        "The Precision-Recall curve is computed by varying the confidence threshold. The area under this "
        "curve is the Average Precision (AP). Since we have a single class (person), AP = mAP. "
        "The @0.5 means a detection is correct if it overlaps ≥ 50% with the ground truth box."
    )

    add_heading(doc, "mAP@0.5:0.95 (77.6%)", level=3)
    add_body(doc,
        '"Strict detection quality across multiple IoU thresholds"\n\n'
        "This is the COCO-standard metric. It averages AP at IoU thresholds [0.50, 0.55, 0.60, ..., 0.95]. "
        "This penalizes loose bounding boxes more severely. A score of 77.6% is strong for a person detector."
    )

    add_heading(doc, "7.3 Inference Speed", level=2)
    add_styled_table(doc,
        ["Metric", "Value"],
        [
            ["Preprocess",    "1.6 ms/image"],
            ["Inference",     "66.1 ms/image"],
            ["Post-process",  "0.8 ms/image"],
            ["Total",         "~68.5 ms/image (~15 FPS)"],
            ["GPU",           "Quadro T2000 (4GB)"],
        ],
        col_widths=[5, 10],
    )

    add_heading(doc, "7.4 Training Progression (Final 3 Epochs)", level=2)
    add_styled_table(doc,
        ["Epoch", "Box Loss", "Cls Loss", "DFL Loss", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
        [
            ["78", "0.679", "0.460", "0.972", "0.921", "0.880", "0.944", "0.772"],
            ["79", "0.679", "0.459", "0.973", "0.919", "0.881", "0.944", "0.773"],
            ["80", "0.680", "0.458", "0.977", "0.924", "0.882", "0.944", "0.776"],
        ],
    )
    add_note_box(doc,
        "The model showed stable convergence in the last epochs with minimal fluctuation, "
        "indicating the 80-epoch training was sufficient and the model did not overfit."
    )

    doc.add_page_break()

    # ──────────────────────── 8. TRAINING CURVES ──────────────────────────

    add_heading(doc, "8. Training Curves & Graphs", level=1)

    add_body(doc, "The following graph shows all training and validation metrics over the 80-epoch training run:")
    add_image_safe(doc, IMG["results"], Inches(6),
        "Figure 2: Training curves — box loss, classification loss, DFL loss, precision, recall, mAP@0.5, mAP@0.5:0.95")

    add_body(doc,
        "Key Observations:\n"
        "• Train Box Loss: Decreased from ~1.3 → ~0.68 — model learned tight bounding boxes.\n"
        "• Train Cls Loss: Decreased from ~1.4 → ~0.46 — strong person/background discrimination.\n"
        "• Val mAP@0.5: Rapidly climbed from 66% → 94.4% and plateaued after epoch ~50.\n"
        "• Val mAP@0.5:0.95: Steady improvement from 40% → 77.6%.\n"
        "• No overfitting: Validation losses continued to decrease alongside training losses."
    )

    add_heading(doc, "Precision-Recall Curve", level=2)
    add_image_safe(doc, IMG["pr_curve"], Inches(5),
        "Figure 3: Precision-Recall curve — 0.944 mAP@0.5 for the person class")

    add_heading(doc, "F1-Confidence Curve", level=2)
    add_image_safe(doc, IMG["f1_curve"], Inches(5),
        "Figure 4: F1-Confidence curve — optimal confidence threshold for precision/recall balance")

    doc.add_page_break()

    # ──────────────────────── 9. CONFUSION MATRIX ─────────────────────────

    add_heading(doc, "9. Confusion Matrix", level=1)

    add_heading(doc, "Normalized Confusion Matrix", level=2)
    add_image_safe(doc, IMG["conf_matrix_n"], Inches(4.5),
        "Figure 5: Normalized confusion matrix showing classification accuracy")

    add_heading(doc, "Raw Confusion Matrix", level=2)
    add_image_safe(doc, IMG["conf_matrix"], Inches(4.5),
        "Figure 6: Raw confusion matrix showing absolute TP, FP, FN counts")

    doc.add_page_break()

    # ──────────────────────── 10. SAMPLE PREDICTIONS ──────────────────────

    add_heading(doc, "10. Sample Test Predictions", level=1)

    add_heading(doc, "Ground Truth Labels", level=2)
    add_image_safe(doc, IMG["val0_labels"], Inches(5.5),
        "Figure 7: Ground truth — Validation Batch 0 (human-annotated bounding boxes)")

    add_heading(doc, "Model Predictions — Batch 0", level=2)
    add_image_safe(doc, IMG["val0_pred"], Inches(5.5),
        "Figure 8: Model predictions — Validation Batch 0 (YOLOv8s with confidence scores)")

    doc.add_page_break()

    add_heading(doc, "Model Predictions — Batch 1", level=2)
    add_image_safe(doc, IMG["val1_pred"], Inches(5.5),
        "Figure 9: Model predictions — Validation Batch 1 (diverse test images)")

    add_heading(doc, "Training Batch (Final Epoch)", level=2)
    add_image_safe(doc, IMG["train_last"], Inches(5.5),
        "Figure 10: Training batch from epoch 80 — augmented images with mosaic, color jitter, transforms")

    doc.add_page_break()

    # ──────────────────────── 11. INFERENCE PIPELINE ──────────────────────

    add_heading(doc, "11. End-to-End Inference Pipeline", level=1)
    add_body(doc,
        "The complete inference pipeline processes camera feeds in real-time through five stages:"
    )
    add_body(doc,
        "Camera Feed → FrameExtractor → IntrusionDetector (YOLOv8s) → ByteTracker → ZoneManager → AlertEngine → Alert Output"
    )

    add_styled_table(doc,
        ["Stage", "Module", "Function", "Latency"],
        [
            ["1. Capture",    "FrameExtractor",     "Threaded RTSP/USB capture → bounded queue",   "~2 ms"],
            ["2. Detect",     "IntrusionDetector",   "YOLOv8s inference on GPU",                    "~66 ms"],
            ["3. Track",      "ByteTracker",         "IoU-based track association across frames",   "~1 ms"],
            ["4. Zone Check", "ZoneManager",         "Point-in-polygon + boundary crossing",        "~0.5 ms"],
            ["5. Alert",      "AlertEngine",         "Enrichment, thumbnail save, Redis publish",   "~1 ms"],
            ["Total",         "",                    "",                                            "~70 ms (~14 FPS)"],
        ],
        col_widths=[2.5, 3.5, 6, 3],
    )

    # ──────────────────────── 12. EDGE DEPLOYMENT ─────────────────────────

    add_heading(doc, "12. Edge Deployment Strategy", level=1)

    add_heading(doc, "12.1 ONNX Export", level=2)
    add_body(doc,
        "The trained PyTorch model is exported to ONNX format for optimized edge inference. "
        "Export settings: opset=17, simplify=True, dynamic=False."
    )

    add_heading(doc, "12.2 Runtime Providers", level=2)
    add_styled_table(doc,
        ["Priority", "Provider", "Hardware"],
        [
            ["1", "CUDAExecutionProvider",  "NVIDIA GPU (Jetson, dGPU)"],
            ["2", "CPUExecutionProvider",   "Fallback for CPU-only devices"],
        ],
        col_widths=[2, 5, 8],
    )

    add_heading(doc, "12.3 Deployment Targets", level=2)
    add_styled_table(doc,
        ["Device", "Expected FPS", "Notes"],
        [
            ["NVIDIA Jetson Nano",       "~8–12 FPS",   "With FP16, limited by 128-core GPU"],
            ["NVIDIA Jetson Xavier NX",  "~25–30 FPS",  "Recommended edge device"],
            ["Quadro T2000",             "~15 FPS",     "Development/testing"],
            ["x86 + RTX GPU",           "~45+ FPS",    "Server-grade deployment"],
        ],
        col_widths=[5, 3, 7],
    )

    doc.add_page_break()

    # ──────────────────────── 13. LIMITATIONS ─────────────────────────────

    add_heading(doc, "13. Limitations & Future Improvements", level=1)

    add_heading(doc, "13.1 Current Limitations", level=2)
    add_styled_table(doc,
        ["Limitation", "Impact", "Severity"],
        [
            ["Single-class only",           "Only detects person — no vehicles, animals, objects",             "Medium"],
            ["Night vision",                "Performance degrades in very low-light / IR cameras",            "High"],
            ["Small persons",               "Persons < 32px at 640px resolution may be missed (~12% gap)",    "Medium"],
            ["Dense crowds",                "Tracking ID switches increase in crowds > 20 persons",           "Medium"],
            ["Static camera assumption",    "Zone polygons are per-camera — PTZ cameras need re-calibration", "Low"],
            ["FP16 on T2000",              "Quadro T2000 has limited FP16 — FP32 required for stable inference","Low"],
        ],
        col_widths=[4.5, 7.5, 3],
    )

    add_heading(doc, "13.2 Recommended Improvements", level=2)
    add_styled_table(doc,
        ["Improvement", "Expected Impact", "Effort"],
        [
            ["Train on full COCO train2017 (118K imgs)",  "+2–3% mAP improvement",                "Low"],
            ["Add thermal/IR training data",              "Night-time performance boost",          "Medium"],
            ["Upgrade to YOLOv8m (larger GPU)",           "+1–2% mAP, better small-person recall", "Low"],
            ["TensorRT optimization",                     "2–3× faster inference on Jetson",       "Medium"],
            ["Re-ID feature extraction",                  "Reduce tracking ID switches in crowds", "High"],
            ["Active learning pipeline",                  "Continuous improvement from deploy data","High"],
        ],
        col_widths=[6, 5, 4],
    )

    doc.add_paragraph()
    add_note_box(doc,
        "Model Weights: runs/train/intrusion_yolov8s_person/weights/best.pt  |  "
        "Inference Config: src/training/config.yaml",
        "📁  FILES"
    )

    # ── Footer paragraph
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("— End of Documentation Report —")
    run.font.size = Pt(10)
    run.italic = True
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    # ── Save ──────────────────────────────────────────────────────────────
    doc.save(OUTPUT_PATH)
    abs_path = os.path.abspath(OUTPUT_PATH)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\n[OK] Report saved to: {abs_path}")
    print(f"     File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    build_report()

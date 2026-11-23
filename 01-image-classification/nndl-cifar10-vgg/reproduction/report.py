"""Build the CIFAR-10 portfolio report from recorded metrics and analysis."""
import argparse
import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from model import VARIANTS

INK = "#172d46"
TEAL = "#146c68"
CODE = (
    "https://github.com/fcxuu/computer-vision-projects/tree/portfolio/reproductions-2026-11-23"
    "/01-image-classification/nndl-cifar10-vgg/reproduction"
)
WEIGHT_BASE = (
    "https://github.com/fcxuu/computer-vision-projects/releases/download/"
    "portfolio-cifar10-vgg-2026-11-23"
)
DATA = "https://www.cs.toronto.edu/~kriz/cifar.html"
PAPER = "https://arxiv.org/abs/1712.09913"
LABEL = {"vgg_a": "VGG-A", "vgg_a_bn": "VGG-A + BN", "vgg_a_dropout": "VGG-A + Dropout"}


def read(path):
    return json.loads(path.read_text())


def figure(path, width, height):
    with PILImage.open(path) as image:
        scale = min(width / image.width, height / image.height)
        return Image(str(path), width=image.width * scale, height=image.height * scale)


def table(rows, widths):
    output = Table(rows, colWidths=widths, hAlign="LEFT")
    output.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf4f6"), colors.white]),
    ]))
    return output


def footer(canvas, doc):
    canvas.setFillColor(colors.HexColor(INK))
    canvas.setFont("Helvetica", 9)
    canvas.drawString(44, 27, "CIFAR-10 / VGG-A and batch normalization")
    canvas.drawRightString(A4[0] - 44, 27, str(doc.page))


def build(run, evidence):
    analysis = read(evidence / "analysis.json")
    full = [run / f"full-{name}" for name in VARIANTS]
    primary = [(read(folder / "run.json"), read(folder / "test.json")) for folder in full]
    half = run / "main-vgg_a"
    half_result = (read(half / "run.json"), read(half / "test.json"))
    results = primary + [half_result]
    winner, winner_test = max(primary, key=lambda trial: (trial[0]["best"]["val_accuracy"],
                                                          -trial[0]["best"]["val_ce"]))
    weight_url = f"{WEIGHT_BASE}/full-{winner['variant']}.pt"
    rows = [["Model", "Params", "Best epoch", "Val acc.", "Test acc.", "Fit time"]]
    for trained, tested in results:
        if trained["best"]["epoch"] != tested["epoch"] or tested["test_samples"] != 10000:
            raise ValueError("Test record does not match the checkpoint protocol")
        name = LABEL[trained["variant"]] + (" (half-width)" if trained["width"] == 0.5 else "")
        rows.append([name, f"{trained['parameters']/1e6:.2f}M", tested["epoch"],
                     f"{trained['best']['val_accuracy']:.2%}", f"{tested['test_accuracy']:.2%}",
                     f"{trained['seconds']/60:.1f} min"])
    reference = read(run / "sweep-vgg_a_bn-0.001" / "run.json")
    probes = [("ReLU + CE + AdamW", reference)] + [
        (name, read(run / folder / "run.json"))
        for name, folder in [("GELU", "probe-gelu"), ("Focal loss", "probe-focal"),
                             ("Weight decay 0.001", "probe-decay"), ("SGD", "probe-sgd")]
    ]
    probe_rows = [["Half-width VGG-A + BN", "Initial LR", "Best val acc.", "Best val CE"]]
    for name, trial in probes:
        probe_rows.append([name, f"{trial['lr']:g}", f"{trial['best']['val_accuracy']:.2%}",
                           f"{trial['best']['val_ce']:.4f}"])
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.textColor = colors.HexColor(INK)
    styles["Title"].fontSize = 25
    styles["Title"].leading = 29
    styles["Title"].alignment = 0
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 8

    def title(text):
        return Paragraph(text, styles["Title"])

    def body(text):
        return Paragraph(text, styles["BodyText"])

    def link(url, name):
        return f'<link href="{url}" color="{TEAL}"><u>{name}</u></link>'

    plain, bn = VARIANTS[:2]
    sweep = analysis["lr_sweep"]
    surfaces = analysis["surfaces"]
    gradients = analysis["gradient_probes"]
    stats = [["Model", "Sweep mean width", "Sweep max width", "Grid CE range", "Max gradient change"]]
    for name in (plain, bn):
        metrics = sweep[name]
        surface = surfaces[name]
        ray = gradients[name]
        stats.append([
            LABEL[name], f"{metrics['mean_width']:.4f}", f"{metrics['max_width']:.4f}",
            f"{surface['maximum'] - surface['minimum']:.4f}", f"{ray['maximum_gradient_difference']:.4f}",
        ])
    story = [
        title("VGG-A and batch normalization on CIFAR-10"),
        body(f"<b>{LABEL[winner['variant']]}</b> achieved "
             f"{winner_test['test_accuracy']:.2%} test accuracy "
             f"({winner_test['test_correct']:,} / {winner_test['test_samples']:,}), corresponding to "
             f"{1 - winner_test['test_accuracy']:.2%} test error."),
        body(f"The experiment uses a fixed stratified split of {winner['train_samples']:,} training images "
             f"and {winner['validation_samples']:,} validation images with seed {winner['seed']}. "
             f"Test metrics use the official held-out 10,000 images. Best validation accuracy chooses weights, "
             f"with lower validation cross-entropy breaking ties."),
        body("The full-width variants share eight 3 x 3 convolutions, five max pools, and a "
             "three-layer classifier. BN adds normalization after each convolution and before the "
             "hidden classifier activations. Dropout adds two classifier masks with probability 0.5. "
             "The half-width model halves channel and hidden-unit counts without changing the topology."),
        body(f"All four comparisons run {winner['epochs']} epochs with batch size {winner['batch']}, "
             "train-only reflected random crop and horizontal flip, AdamW, initial LR 0.001, "
             "weight decay 0.0001, and cosine decay. ReLU and cross-entropy are held fixed."),
        table(rows, [147, 55, 67, 64, 66, 86]), Spacer(1, 13),
        body(f"The short optimization probes train half-width VGG-A + BN for {reference['epochs']} epochs "
             "at each setting. The focal loss uses gamma 2. Validation uses CE for all settings. "
             "The SGD probe also changes the learning rate, so the comparison concerns whole configurations."),
        table(probe_rows, [201, 82, 105, 97]), Spacer(1, 13),
        body(f"{link(DATA, 'Dataset')} &nbsp;&nbsp; {link(CODE, 'Code and all checkpoints')} "
             f"&nbsp;&nbsp; {link(weight_url, 'Download selected weights')} "
             f"&nbsp;&nbsp; {link(PAPER, 'Loss landscape reference')}"),
        PageBreak(), title("Learning curves and filters"),
        figure(evidence / "curves.png", 503, 255),
        body("Training losses come from augmented batches at each update. Validation uses clean images. "
             "The full-width comparison holds the width, data, optimizer and schedule fixed."),
        figure(evidence / "filters.png", 503, 325),
        body("Each tile is a learned RGB kernel from the first convolution. Color contrast represents signed "
             "coefficients after symmetric scaling. These are local weights at the input, before any BN layer."),
        PageBreak(), title("Learning-rate sensitivity"),
        body(f"Matched half-width VGG-A and VGG-A + BN runs cover {sweep[plain]['steps']:,} "
             "updates per rate. The four fixed rates are 0.0001, 0.0005, 0.001 and 0.002. "
             "The min/max loss envelope compares training trajectories at the same update. "
             "Gradient norms are measured before the update."),
        figure(evidence / "lr-sweep.png", 503, 480),
        body(f"Mean envelope width is {sweep[plain]['mean_width']:.4f} without BN and "
             f"{sweep[bn]['mean_width']:.4f} with BN. The envelope reflects sensitivity to these "
             "rates and the resulting parameter trajectories."),
        PageBreak(), title("Local loss and gradient measurements"),
        body(f"At the best full-width checkpoints, the surface uses {surfaces[plain]['samples']} "
             "fixed validation images. Two random directions match each filter's L2 norm. "
             "Biases, BN affine parameters and running statistics remain fixed."),
        figure(evidence / "surface.png", 503, 220),
        body(f"The gradient probe uses {gradients[plain]['samples']} fixed validation images. It moves along "
             "the negative center gradient and compares observed loss with the first-order prediction. "
             "Gradient change is measured over the sampled ray only."),
        figure(evidence / "gradient-probe.png", 503, 225),
        table(stats, [101, 95, 91, 88, 110]), Spacer(1, 8),
        body("In this run the BN checkpoint has a smaller grid CE range and smaller maximum gradient "
             "change on the sampled ray, while its across-rate training envelope is wider. At radius "
             "0.03 both losses rise even though the linear prediction decreases. The two diagnostics "
             "measure different aspects of optimization on this split."),
    ]
    doc = SimpleDocTemplate(str(evidence / "report.pdf"), pagesize=A4, leftMargin=44, rightMargin=44,
                            topMargin=40, bottomMargin=49, title="CIFAR-10 VGG-A and BatchNorm")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    build(args.run, args.evidence)


if __name__ == "__main__":
    main()

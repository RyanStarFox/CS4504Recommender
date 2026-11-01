"""Rebuild B's figures, three contribution slides and speaker notes from reports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

matplotlib.rcParams["svg.hashsalt"] = "recommender-b"
REPORTS = Path("data/text_sasrec")
FIGURES = Path("docs/figures")
FONT = "Microsoft YaHei"
NAVY, TEAL, ORANGE, GREY = "142B43", "12857B", "D77A32", "526579"


def load_report(name):
    return json.loads((REPORTS / f"{name}.json").read_text())


def plot_metrics(reports, labels, colors, name):
    fig, ax = plt.subplots(figsize=(6.5, 3.9), layout="constrained")
    x = np.arange(2)
    width = 0.31
    for i, (report, label, color) in enumerate(zip(reports, labels, colors)):
        values = [report["test_metrics"][key] for key in ["hr10", "ndcg10"]]
        bars = ax.bar(
            x + (i - 0.5) * width, values, width, label=label, color="#" + color
        )
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.set_xticks(x, ["HR@10", "NDCG@10"])
    ax.set_ylim(0, max(r["test_metrics"]["hr10"] for r in reports) * 1.3)
    ax.set_ylabel("Test metric (all 6,040 users)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper right")
    fig.supxlabel(
        "One fixed seed (42); validation-only checkpoint selection",
        fontsize=8,
        color="#526579",
    )
    for extension in ["png", "svg"]:
        output = FIGURES / f"{name}.{extension}"
        fig.savefig(
            output,
            dpi=200,
            bbox_inches="tight",
            metadata={"Date": None} if extension == "svg" else {},
        )
        if extension == "svg":
            output.write_text(
                "\n".join(line.rstrip() for line in output.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def text(slide, x, y, w, h, content, size=18, color="1A1A1A", bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate(content.split("\n")):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = RGBColor.from_string(color)
        paragraph.space_after = Pt(8)
        paragraph.line_spacing = 1.15
    return box


def slide_base(prs, section, title, question):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
    text(slide, 0.63, 0.43, 12.0, 0.30, section, 14, "1F3A68")
    text(slide, 0.63, 0.79, 12.0, 0.65, title, 32, "1F3A68", True)
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.63), Inches(1.49), Inches(3.15), Inches(0.02)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = RGBColor.from_string("D5DAE3")
    rule.line.fill.background()
    text(slide, 0.63, 1.68, 12.0, 0.70, question, 18)
    return slide


def metrics_table(slide, y, rows):
    table = slide.shapes.add_table(
        len(rows) + 1,
        3,
        Inches(6.95),
        Inches(y),
        Inches(5.70),
        Inches(0.55 * (len(rows) + 1)),
    ).table
    for column, width in zip(table.columns, [3.0, 1.35, 1.35]):
        column.width = Inches(width)
    all_rows = [["模型", "HR@10", "NDCG@10"], *rows]
    for i, row in enumerate(all_rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = value
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(
                "1F3A68" if i == 0 else ("FFFFFF" if i % 2 else "F4F6F9")
            )
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            para.font.name = FONT
            para.font.size = Pt(15)
            para.font.bold = i == 0
            para.font.color.rgb = RGBColor.from_string("FFFFFF" if i == 0 else "1A1A1A")
    return table


def metric_row(label, report):
    metrics = report["test_metrics"]
    return [label, f"{metrics['hr10']:.6f}", f"{metrics['ndcg10']:.6f}"]


def main():
    import csv

    FIGURES.mkdir(parents=True, exist_ok=True)
    gpt = load_report("sasrec_gpt2_gated")
    qwen = load_report("sasrec_qwen_gated")
    aligned = load_report("sasrec_qwen_aligned")
    base = dict(aligned["experiment_config"])
    base.pop("alignment")
    assert base == qwen["experiment_config"]
    plot_metrics(
        [gpt, qwen], ["GPT-2 + gate", "Qwen + gate"], [GREY, TEAL], "b-encoders"
    )
    plot_metrics(
        [qwen, aligned], ["Qwen + gate", "+ alignment"], [TEAL, ORANGE], "b-alignment"
    )
    with Path("data/results.csv").open(newline="", encoding="utf-8") as stream:
        id_row = next(
            row for row in csv.DictReader(stream) if row["model"] == "sasrec_id"
        )
    delta_hr = aligned["test_metrics"]["hr10"] - qwen["test_metrics"]["hr10"]
    delta_ndcg = aligned["test_metrics"]["ndcg10"] - qwen["test_metrics"]["ndcg10"]
    relative_ndcg = (
        aligned["test_metrics"]["ndcg10"] / qwen["test_metrics"]["ndcg10"] - 1
    ) * 100
    hits = round(delta_hr * 6040)
    prs = Presentation()
    # Match the submitted ranking deck's canvas, type, colors and table style.
    reference = Presentation("docs/c-ranking.pptx")
    prs.slide_width, prs.slide_height = reference.slide_width, reference.slide_height
    prs.core_properties.title = "电影文本表示、门控融合与对比对齐"
    prs.core_properties.subject = "序列推荐中的文本表示对照实验"

    slide = slide_base(
        prs,
        "电影文本表示与序列推荐",
        "电影文本表示与门控融合",
        "在电影ID表示之外引入标题和类型，比较GPT-2与Qwen提取的文本特征。",
    )
    text(slide, 0.63, 2.55, 5.95, 0.4, "电影文本编码", 19, "1F3A68", True)
    text(
        slide,
        0.63,
        2.98,
        5.95,
        0.90,
        "用 GPT-2 或 Qwen3.5-0.8B 编码电影标题和类型，编码器不更新。",
    )
    text(slide, 0.63, 3.95, 5.95, 0.4, "门控融合", 19, "1F3A68", True)
    text(
        slide,
        0.63,
        4.38,
        5.95,
        0.90,
        "将文本向量映射到64维，由门控学习文本权重，再与电影ID向量相加。",
    )
    text(slide, 0.63, 5.35, 5.95, 0.4, "下一部电影预测", 19, "1F3A68", True)
    text(
        slide,
        0.63,
        5.78,
        5.95,
        0.65,
        "融合表示输入 SASRec（自注意力序列模型），用于预测用户下一次交互的电影。",
        16,
    )
    metrics_table(
        slide,
        2.58,
        [
            [
                "仅交互ID的 SASRec*",
                f"{float(id_row['hr10']):.6f}",
                f"{float(id_row['ndcg10']):.6f}",
            ],
            metric_row("SASRec + GPT-2文本", gpt),
            metric_row("SASRec + Qwen文本", qwen),
        ],
    )
    text(
        slide,
        6.95,
        4.98,
        5.7,
        0.65,
        "HR@10：前10名命中真实下一次交互的比例。\nNDCG@10：命中后位置越靠前，得分越高。",
        14,
        "5C6370",
    )
    text(
        slide,
        6.95,
        5.78,
        5.7,
        0.65,
        "*纯ID结果的原始训练配置未保存，与文本模型的比较仅作参考。",
        14,
        "5C6370",
    )
    text(
        slide,
        0.63,
        6.68,
        12,
        0.45,
        "GPT-2与Qwen各有高低，本次实验没有显示出文本融合的优势。",
        17,
        "1F3A68",
    )
    text(
        slide,
        0.63,
        7.17,
        12,
        0.25,
        "MovieLens-1M；6040位用户、3883部电影；按时间划分；两种文本模型共用训练配置。",
        12,
        "5C6370",
    )
    slide.notes_slide.notes_text_frame.text = (
        "这里我们在电影ID向量之外增加了标题和类型。先用预训练语言模型抽取文本向量，"
        "再投影到64维，通过门控与ID向量相加。语言模型不参与训练，推荐模型和门控参数由交互数据学习。"
        "SASRec用这组表示编码历史，并给候选电影评分。\n\n"
        "两种文本模型使用相同的数据划分和训练设置。GPT-2的NDCG略高，Qwen的命中率略高，"
        "整体接近，都没有超过表中的纯ID结果。纯ID原始训练配置没有保存，所以这里先保留结果对照，"
        "不能把差距完全归因于文本。测试覆盖全部6040位用户，按时间留出真实下一次交互。"
    )

    slide = slide_base(
        prs,
        "电影语义与行为表示对齐",
        "电影行为表示与文本表示的对齐",
        "增加对比损失，使同一电影的两种表示接近，并与其他电影区分。",
    )
    text(slide, 0.63, 2.55, 5.95, 0.4, "正负样本", 19, "1F3A68", True)
    text(
        slide,
        0.63,
        2.98,
        5.95,
        1.05,
        "正例：同一电影的ID向量和文本向量。\n负例：该电影的ID向量与其他电影的文本向量。",
    )
    text(
        slide,
        0.63,
        4.12,
        5.95,
        0.5,
        "训练损失 = 排序损失 + 0.01 × 对齐损失",
        17,
        "1F3A68",
    )
    text(
        slide,
        0.63,
        4.70,
        5.95,
        0.75,
        "温度0.07，每批最多256部电影。\n两组都冻结语言模型，不使用用户属性。",
        16,
    )
    text(slide, 0.63, 5.65, 5.95, 0.4, "实验设置", 19, "1F3A68", True)
    text(
        slide,
        0.63,
        6.08,
        5.95,
        0.65,
        "模型、划分、种子和训练预算均与未对齐版本相同。",
        16,
    )
    metrics_table(
        slide,
        2.58,
        [
            metric_row("Qwen文本融合", qwen),
            metric_row("Qwen文本融合 + 对齐", aligned),
        ],
    )
    text(
        slide,
        6.95,
        4.58,
        5.7,
        0.85,
        f"命中用户 {hits:+d} / 6040；HR@10 {delta_hr:+.6f}\nNDCG@10 {delta_ndcg:+.6f}（相对{relative_ndcg:+.2f}%）",
        16,
        "1F3A68",
    )
    text(
        slide,
        6.95,
        5.65,
        5.7,
        0.90,
        "两个指标略有提高，但只运行了一个种子，增益是否稳定还需重复实验。",
        16,
        "5C6370",
    )
    text(
        slide,
        0.63,
        7.02,
        12,
        0.35,
        "本次对齐模型略好于未对齐模型，但仍低于表中的纯ID基线。",
        16,
        "1F3A68",
    )
    slide.notes_slide.notes_text_frame.text = (
        "接着固定Qwen编码器，只增加电影级对齐。我们把同一电影的ID向量和投影后的文本向量配成正例，"
        "其他电影作为负例。排序损失仍然是BPR，对齐损失是双向InfoNCE，权重0.01、温度0.07在运行前固定。\n\n"
        "两个版本都从相同初始化开始，采用相同负采样、批次和早停设置。按验证集NDCG选出最佳参数后，"
        f"增加对齐的模型在6040位测试用户中净增{hits}位命中，NDCG绝对增加{delta_ndcg:.6f}，"
        f"相对约{relative_ndcg:.2f}%。变化不大，目前只跑了一个种子，因此我们把它看作一次小幅改善，"
        "还需重复实验确认。"
    )

    slide = slide_base(
        prs,
        "电影文本表示部分",
        "工作小结",
        "完成电影文本编码、门控融合和对比对齐，并比较不同设置下的推荐效果。",
    )
    text(slide, 0.63, 2.55, 5.95, 0.4, "主要工作", 19, "1F3A68", True)
    text(
        slide,
        0.63,
        3.02,
        5.95,
        3.10,
        "提取GPT-2和Qwen的冻结电影特征。\n"
        "实现文本投影与门控，接入SASRec。\n"
        "加入电影级对比损失，完成对齐对照。\n"
        "保存训练配置、指标和检查点。",
        18,
    )
    text(slide, 6.95, 2.55, 5.7, 0.4, "实验发现", 19, "1F3A68", True)
    text(
        slide,
        6.95,
        3.02,
        5.7,
        3.10,
        "GPT-2和Qwen的结果接近，各有高低。\n"
        "对齐后两个指标略升，仍需重复实验。\n"
        "目前表现最好的仍是已有纯ID模型。",
        18,
    )
    text(
        slide,
        0.63,
        6.65,
        12,
        0.65,
        "后续需补同配置纯ID基线和多种子实验；当前只使用了标题与类型两类文本。",
        16,
        "5C6370",
    )
    slide.notes_slide.notes_text_frame.text = (
        "我负责文本表示这一部分：从电影元数据抽取冻结特征，把投影和门控接入推荐模型，"
        "再完成对齐实验。实验配置和训练记录都已保存。当前结果里，加入文本没有超过已有纯ID模型。"
        "接下来应先补齐同配置基线和重复实验，确认这种差异来自方法本身还是训练设置。"
    )
    prs.save("docs/b-representation.pptx")
    notes = "# 电影文本表示与对比对齐：讲稿\n\n"
    for slide in prs.slides:
        title = next(
            shape.text
            for shape in slide.shapes
            if shape.has_text_frame
            and shape.text_frame.paragraphs[0].font.size == Pt(32)
        )
        notes += f"## {title}\n\n{slide.notes_slide.notes_text_frame.text}\n\n"
    Path("docs/b-speaker-notes.md").write_text(notes.rstrip() + "\n", encoding="utf-8")
    print(
        "Rebuilt self-contained representation slides and notes in the ranking deck's style"
    )


if __name__ == "__main__":
    main()

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

matplotlib.rcParams["svg.hashsalt"] = "recommender-b"
REPORTS = Path("data/text_sasrec")
FIGURES = Path("docs/figures")
FONT = "WenQuanYi Zen Hei"
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


def text(slide, x, y, w, h, lines, size=22, color=NAVY, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate(lines.split("\n")):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = RGBColor.from_string(color)
        paragraph.space_after = Pt(12)
    return box


def slide_base(prs, page, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string("F5F8FA")
    text(slide, 0.55, 0.3, 1.8, 0.45, f"B  /  {page}", 15, TEAL, True)
    text(slide, 0.55, 0.95, 12.2, 0.8, title, 32, NAVY, True)
    text(slide, 0.55, 1.8, 12.1, 0.6, subtitle, 16, GREY)
    return slide


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    gpt = load_report("sasrec_gpt2_gated")
    qwen = load_report("sasrec_qwen_gated")
    aligned = load_report("sasrec_qwen_aligned")
    base = dict(aligned["experiment_config"])
    options = base.pop("alignment")
    assert base == qwen["experiment_config"]
    plot_metrics(
        [gpt, qwen], ["GPT-2 + gate", "Qwen + gate"], [GREY, TEAL], "b-encoders"
    )
    plot_metrics(
        [qwen, aligned], ["Qwen + gate", "+ alignment"], [TEAL, ORANGE], "b-alignment"
    )
    delta_hr = aligned["test_metrics"]["hr10"] - qwen["test_metrics"]["hr10"]
    delta_ndcg = aligned["test_metrics"]["ndcg10"] - qwen["test_metrics"]["ndcg10"]
    conclusion = f"相对 5b：HR@10 {delta_hr:+.5f}；NDCG@10 {delta_ndcg:+.5f}。"
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    slide = slide_base(
        prs,
        "05",
        "电影文本表示与门控融合",
        "冻结编码器，只训练推荐模型、电影投影和门控；不使用用户属性文本。",
    )
    text(
        slide,
        0.6,
        2.65,
        5.35,
        3.6,
        "标题 + 类型 → 最后一层掩码平均池化\n"
        "GPT-2：768 维；Qwen：1024 维\n"
        "3883 部电影，按原始 ID 缓存 float32\n"
        "投影 + LayerNorm → 可训练门控 → ID 向量\n"
        "历史编码与候选打分共用融合表示",
        20,
    )
    slide.shapes.add_picture(
        str(FIGURES / "b-encoders.png"), Inches(6.15), Inches(2.55), width=Inches(6.55)
    )
    text(
        slide,
        0.6,
        6.65,
        12,
        0.5,
        "5a/5b 使用相同划分、种子、SASRec 初始化与训练预算；单种子下没有一致胜出的编码器。",
        15,
        GREY,
    )
    slide.notes_slide.notes_text_frame.text = (
        "这页只比较冻结电影编码器。电影文本只有标题和类型；缓存按原始电影编号索引，"
        "padding 和缺失行保持零值。推荐维度独立设为64，通过投影和门控注入。"
        "GPT-2 的测试 NDCG 略高，Qwen 的 HR 略高，不能预设换大模型必然提升。"
        "表中随机切分的课程 NCF 不与这些时间切分结果直接比较。"
    )
    slide = slide_base(
        prs,
        "06",
        "对比对齐：5b 与第 6 行",
        "保留相同模型与训练协议，只增加训练电影 ID 和投影文本之间的对齐目标。",
    )
    text(
        slide,
        0.6,
        2.65,
        5.3,
        3.6,
        "同一电影的 ID 向量 ↔ 投影文本向量\n"
        "对称 InfoNCE，排除 0、缺失与重复 ID\n"
        f"每批最多 {options['max_items']} 部训练电影\n"
        f"总损失：BPR + {options['weight']} × 对齐损失\n"
        f"温度 {options['temperature']}，不查看测试标签调参",
        20,
    )
    slide.shapes.add_picture(
        str(FIGURES / "b-alignment.png"), Inches(6.15), Inches(2.55), width=Inches(6.55)
    )
    text(
        slide,
        0.6,
        6.55,
        12,
        0.7,
        conclusion + " 单种子结果，不作显著性结论。",
        16,
        GREY,
    )
    slide.notes_slide.notes_text_frame.text = (
        "先说明公平性：第6行从相同种子重新训练，不从5b的最佳检查点继续训练；"
        "划分、初始化、负采样、批次、优化器、早停均相同。对齐只读取训练历史与训练正样本，"
        "电影去重后用独立随机数发生器抽取最多256部，不影响推荐模型的dropout随机数。"
        "固定权重0.01和温度0.07，没有按测试结果扫参。"
        + conclusion
        + "这次结果只能说明这一固定配置的观察，不能作显著性或普遍收益结论。"
    )
    slide = slide_base(
        prs,
        "10",
        "B 的贡献与交付",
        "负责冻结文本表示、共享设备策略、门控融合和电影级对比对齐。",
    )
    text(
        slide,
        0.8,
        2.65,
        11.7,
        3.7,
        "冻结表示：交付 GPT-2 / Qwen 电影特征、原始 ID 对齐与生成配置。\n"
        "推荐融合：实现共享投影与门控，完成 5a/5b 的同配置对照。\n"
        "对比对齐：完成第 6 行及与 5b 的消融，结果与配置进入共享表。\n"
        "工程验证：设备选择、梯度、无效 ID、检查点恢复和零权重回归。\n"
        "用户文本为可选消融，本次未启用；未修改其他成员的训练代码。",
        23,
    )
    slide.notes_slide.notes_text_frame.text = (
        "我负责B部分：把两种冻结语言模型转换为可复用电影表示，建立原始ID索引，"
        "实现与SASRec共用的门控接口，再完成电影ID和文本表示的对比对齐实验。"
        "训练配置、结果、图与复现代码均已交付；语言模型始终冻结。"
    )
    prs.save("docs/b-representation.pptx")
    notes = "# B 幻灯片讲稿\n\n"
    for i, slide in enumerate(prs.slides):
        notes += (
            f"## 第 {[5, 6, 10][i]} 页\n\n{slide.notes_slide.notes_text_frame.text}\n\n"
        )
    Path("docs/b-speaker-notes.md").write_text(notes.rstrip() + "\n", encoding="utf-8")
    print("Generated two PNG/SVG figures, B slides 5/6/10 and speaker notes")


if __name__ == "__main__":
    main()

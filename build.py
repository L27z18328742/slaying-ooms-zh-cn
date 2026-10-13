#!/usr/bin/env python3
"""Build the local reading site and the complete Markdown translation (stdlib only)."""
from pathlib import Path
import html
import json
import re

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "webapp"
SOURCE = "https://christianjmills.com/posts/mastering-llms-course-notes/conference-talk-012/"
REVISION = "4fe589582bbe57301cd1046764504686f085e64a"
LICENSE = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
CHAPTERS = [
    ("ch00", "导论", "用 PyTorch FSDP 和 torchao 解决显存不足", "Introduction", "演讲背景、全文目录与原始演讲资料。",
     ["contents", "introduction"]),
    ("ch01", "第 1 章", "理解模型的内存占用", "Understanding Model Memory", "参数、梯度、优化器状态与激活的内存开销。",
     ["understanding-model-memory"]),
    ("ch02", "第 2 章", "优化内存占用", "Optimizing Memory Usage", "参数量化、torch.compile、LoRA、QLoRA 与张量子类。",
     ["optimizing-memory-usage", "optimizer-memory", "parameter-quantization", "gradient-optimization-with-lora", "challenges-and-solutions-with-qlora"]),
    ("ch03", "第 3 章", "使用 FSDP 实现模型并行", "Model Parallelism with FSDP", "数据并行、分片、FSDP1 / FSDP2 与 CPU 卸载。",
     ["model-parallelism-with-fsdp", "data-parallism", "fsdp", "fsdp1-vs.-fsdp2", "cpu-offloading"]),
    ("ch04", "第 4 章", "FSDP2 基准测试", "Benchmarking FSDP2", "双 RTX 3090 实测、配置差异、根因与五类性能差距。",
     ["benchmarking-fsdp2", "benchmarking-plan", "initial-benchmarking-and-discrepancies", "identifying-the-root-causes", "addressing-the-discrepancies-and-further-benchmarking", "deep-dive-into-performance-gaps"]),
    ("references", "参考资源", "来源、术语与翻译说明", "Sources & Translation Notes", "底本版本、翻译范围、术语对照与许可说明。",
     ["source-version", "translation-scope", "terminology", "translation-notes", "licenses"]),
]


def anchored(md, ids):
    headings = re.findall(r"^#{2,3} .+$", md, re.M)
    if len(headings) != len(ids):
        raise ValueError(f"Heading count mismatch: {len(headings)} != {len(ids)}")
    anchors = iter(ids)
    return re.sub(r"^#{2,3} .+$", lambda m: f'<a id="{next(anchors)}"></a>\n\n{m[0]}', md, flags=re.M)


def page(title, md, nav, prefix):
    chapter_links = "".join(
        f'<li><a href="{c[0]}.html">{c[1]} · {c[2]}</a></li>' for c in CHAPTERS
    )
    return f'''<!DOCTYPE html>
<html lang="zh-Hans">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — Slaying OOMs 中文版</title>
<link rel="stylesheet" href="{prefix}assets/style.css"></head>
<body>
<header class="topbar"><a class="brand" href="{prefix}index.html">Slaying OOMs 中文版<span>PyTorch FSDP 与 torchao</span></a>
<nav class="chapnav"><a href="{prefix}all.html">全文阅读</a><a href="{SOURCE}">英文原文 ↗</a></nav></header>
<div class="layout"><aside class="sidebar"><nav class="toc" id="toc"><h2>章节导航</h2>
<ol class="book-toc">{chapter_links if prefix else chapter_links.replace('href="', 'href="chapters/')}</ol><h2>本页目录</h2></nav></aside>
<main class="main"><article id="content" class="prose"></article>
<noscript>此页需要 JavaScript 排版。也可直接阅读目录中的完整译文 Markdown 文件。</noscript>
<nav class="pager">{nav}</nav>
<footer class="pagefoot">原作者 Christian Mills · 2024-07-24 · 非官方中文译本 ·
依据历史版本 {REVISION[:7]} 制作 · 译文 <a href="{LICENSE}">CC BY-NC-SA 4.0</a></footer></main></div>
<script type="application/json" id="chapter-data">{json.dumps(md, ensure_ascii=False).replace('<', chr(92) + 'u003c')}</script>
<script src="{prefix}assets/marked.min.js"></script><script src="{prefix}assets/app.js"></script>
</body></html>
'''


def main():
    (WEB / "chapters").mkdir(parents=True, exist_ok=True)
    translations = []
    for i, c in enumerate(CHAPTERS):
        name, number, title, _, _, ids = c
        md = anchored((ROOT / "content" / f"{name}.md").read_text(), ids)
        translations.append(md)
        links = []
        for offset, label, cls in [(-1, "← 上一章", "prev"), (1, "下一章 →", "next")]:
            j = i + offset
            if 0 <= j < len(CHAPTERS):
                other = CHAPTERS[j]
                links.append(f'<a class="{cls}" href="{other[0]}.html"><span class="dir">{label}</span><div class="ttl">{other[1]}　{other[2]}</div></a>')
        rendered = md.replace("../webapp/assets/", "../assets/")
        (WEB / "chapters" / f"{name}.html").write_text(page(f"{number}　{title}", rendered, "".join(links), "../"))

    combined = "\n\n---\n\n".join(translations)
    combined = re.sub(r"\((?:ch\d+|references)\.html#", "(#", combined)
    (ROOT / "slaying_ooms_zh.md").write_text(combined.replace("../webapp/assets/", "webapp/assets/"))
    (WEB / "all.html").write_text(page("完整译文", combined.replace("../webapp/assets/", "assets/"), "", ""))

    cards = "\n".join(f'''<a class="card" href="chapters/{c[0]}.html"><span class="no">{c[1]}</span><span class="zh">{c[2]}</span><span class="en">{c[3]}</span><span class="desc">{c[4]}</span></a>''' for c in CHAPTERS)
    (WEB / "index.html").write_text(f'''<!DOCTYPE html>
<html lang="zh-Hans"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slaying OOMs 中文版 — PyTorch FSDP 与 torchao</title><link rel="stylesheet" href="assets/style.css"></head>
<body><section class="hero"><div class="eyebrow">简体中文全译本 · 本地阅读版</div>
<h1>用 PyTorch FSDP 和 torchao<br>解决显存不足</h1>
<p class="sub">Slaying OOMs with PyTorch FSDP and torchao<br>Christian Mills 的第 12 场会议演讲笔记 · 讲者 Mark Saroufim、Jane Xu · 2024 年 7 月 24 日</p>
<p class="sub">从模型内存占用、量化与 QLoRA，到 FSDP1 / FSDP2、CPU 卸载，以及双 RTX 3090 基准测试的性能排查。</p>
<p class="reading-actions"><a class="primary-link" href="all.html">阅读完整译文 →</a><a href="chapters/ch04.html#benchmarking-fsdp2">直达 FSDP2 基准测试 →</a></p>
<div class="meta"><div><b>5</b>主要章节</div><div><b>13</b>原文小节</div><div><b>5</b>原始插图</div><div><b>3</b>代码块</div></div></section>
<div class="grid-wrap"><h2>章节与参考资料</h2><div class="chapter-grid">{cards}</div></div>
<footer class="foot">非官方中文译本 · 原作者 <a href="{SOURCE}">Christian Mills</a> · 内容依据历史版本 {REVISION[:7]} · <a href="{LICENSE}">CC BY-NC-SA 4.0</a><br>阅读界面沿用 Ultra-Scale Playbook 中文版的布局与样式。性能结论保留 2024 年语境。</footer>
</body></html>''')
    print(f"Built {len(CHAPTERS)} chapter/resource pages, index.html, all.html and slaying_ooms_zh.md")


if __name__ == "__main__":
    main()

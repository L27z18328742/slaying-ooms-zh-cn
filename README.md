# Slaying OOMs 中文版 — PyTorch FSDP 与 torchao

> **非官方中文译本**。原作者 Christian Mills，演讲者 Mark Saroufim、Jane Xu，原文发表于 2024 年 7 月 24 日。
> [原文](https://christianjmills.com/posts/mastering-llms-course-notes/conference-talk-012/) · [底本版本](https://github.com/cj-mills/christianjmills/blob/4fe589582bbe57301cd1046764504686f085e64a/posts/mastering-llms-course-notes/conference-talk-012/index.md) · 译文许可：CC BY-NC-SA 4.0。

按照相邻 `ultrascale-playbook-zh-tw` 目录的格式制作：简体中文、逐章 Markdown、静态阅读页、侧栏目录、前后章导航、原图、代码块与表格。

## 阅读

直接打开 `webapp/index.html`；也可运行：

```bash
bash run.sh
```

随后访问 <http://127.0.0.1:8644>。可用 `bash run.sh 8650` 指定其他端口。默认仅监听本机。

- **全文**：[完整译文 Markdown](slaying_ooms_zh.md) / [HTML](webapp/all.html)。
- **用户指定小节**：[FSDP2 基准测试](webapp/chapters/ch04.html#benchmarking-fsdp2)。
- **目录与来源**：[翻译说明](content/references.md)。

## 目录结构

```text
├── content/                # 5 个主要章节 + 来源、术语与翻译说明
├── webapp/                 # 可直接打开的离线静态阅读页
│   ├── index.html          # 章节目录首页
│   ├── all.html            # 完整译文
│   ├── chapters/           # 章节页
│   └── assets/             # 样式、渲染器、5 张原图
├── src/                    # 固定版本英文底本与来源清单
├── licenses/               # 内容、代码和模板许可
├── slaying_ooms_zh.md       # 完整中文译文（构建生成）
├── NOTICE.md               # 署名及修改说明
├── VALIDATION.md           # 核对结果
├── build.py                # 仅依赖 Python 标准库
└── run.sh                  # 本地阅读服务
```

修改 `content/*.md` 后，运行 `python3 build.py` 重建。页面资源均在本地；原文链接、资料和 trace 文件需要联网。

## 覆盖范围

完整翻译底本的 5 个主要章节、13 个小节，保留 5 张原图、3 段代码、1 张基准表和资料链接。图片内英文标签保留，图题已翻译。原网页后来添加的作者推广卡片与站点导航不属于该历史底本。

所有性能描述均保留 **2024 年**的语境。本次没有运行训练或声称重新验证性能。译者补充均明确标注。原文及许可绑定具体提交；不将历史许可套用于当前站点后来添加的内容。

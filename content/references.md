# 参考资源　来源、术语与翻译说明

## 原文与版本（Source Version）

- **标题**：Conference Talk 12: Slaying OOMs with PyTorch FSDP and torchao。
- **作者**：Christian Mills。
- **演讲者**：Mark Saroufim、Jane Xu。
- **发布日期**：2024 年 7 月 24 日。
- **原网页**：[文章入口](https://christianjmills.com/posts/mastering-llms-course-notes/conference-talk-012/)。
- **用户指定小节**：[Benchmarking FSDP2](https://christianjmills.com/posts/mastering-llms-course-notes/conference-talk-012/#benchmarking-fsdp2)。
- **翻译底本**：[Git 提交 `4fe589582bbe57301cd1046764504686f085e64a` 中的原文 Markdown](https://github.com/cj-mills/christianjmills/blob/4fe589582bbe57301cd1046764504686f085e64a/posts/mastering-llms-course-notes/conference-talk-012/index.md)。
- **该版本许可**：[LICENSE.txt](https://github.com/cj-mills/christianjmills/blob/4fe589582bbe57301cd1046764504686f085e64a/LICENSE.txt)，CC BY-NC-SA 4.0。
- **翻译日期**：2026 年 9 月 20 日。

## 翻译范围（Translation Scope）

译文覆盖底本的全部技术正文：5 个主要章节、13 个小节、5 张插图及其图题、3 个代码块、1 张基准数据表，以及系列说明和演讲资料入口。按照参考目录拆分为逐章 Markdown，并提供完整译文和静态 HTML 阅读页。

图片保留原图，图题译为中文，图片内的英文标签保留。网页导航、评论组件与当前网站后来添加的作者推广卡片不属于本次历史底本的技术正文。原始资料中的外部链接予以保留；访问这些资源仍需要联网。

原文采用提纲式笔记，译文保留原有层次与顺序。页面增加目录、前后章导航与来源说明，不增加原文没有的交互实验。完整译文和章节页都保留英文小节锚点。

## 术语对照（Terminology）

| 英文 | 本译文用语 |
|---|---|
| Out of Memory / OOM | 内存不足；GPU 场景下也称显存不足 |
| VRAM | 显存 |
| parameters / weights | 参数／权重 |
| gradients | 梯度 |
| optimizer state | 优化器状态 |
| activations | 激活 |
| quantization / dequantization | 量化／反量化 |
| scale / scaling factor | 缩放因子 |
| Low-Rank Adaptation / LoRA | 低秩适配 |
| double quantization | 双重量化 |
| tensor subclass | 张量子类 |
| Fully Sharded Data Parallel / FSDP | 全分片数据并行 |
| sharding | 分片 |
| all-gather | all-gather，按上下文解释为收集分片数据 |
| wrapping policy | 包装策略 |
| CPU offloading | CPU 卸载 |
| distributed tensor / DTensor | 分布式张量 |
| dispatch | 分派 |
| fused Adam optimizer kernel | 融合 Adam 优化器内核 |
| rotary positional embeddings / RoPE | 旋转位置编码 |
| scaled dot-product attention / SDPA | 缩放点积注意力 |
| trace | trace／执行轨迹 |
| bnb | bitsandbytes |

## 翻译与编辑说明（Translation Notes）

- 使用简体中文，与参考目录的实际内容一致；目录名中的 `zh-tw` 不作为译文字体依据。
- Python 示例原样保留。Shell 示例只添加续行符，参数及其取值不变。
- 保留所有基准数字以及 GB / GiB 的区别，没有重新进行 GPU 性能测试。
- 原文的 `Data Parallism`、`multipls` 等拼写问题不带入中文；数据并行小节的链接仍使用原始 `data-parallism` 锚点。
- 规范人名 Tim Dettmers 与术语 DTensor、RoPE 的拼写。
- 对 INT8 示例与 4 位量化的区别、64 个参数的含义、优化器局部加速等歧义单独标注译者注。
- 原文损坏的 `nf4tensor.py` 链接按历史引用保留并提示，未虚构一个可用的新地址。
- 原文关于后续版本的计划、性能瓶颈和实现限制均属于 2024 年上下文，未改写成当前产品状态。

## 许可与署名（Licenses）

本文采用已发布的历史版本作为底本，该提交中的内容许可为 **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International（CC BY-NC-SA 4.0）**。译文作为非官方中文改编，采用相同许可，并保留原作者、原作链接和修改说明。作者当前仓库的内容许可与此历史版本不同，请勿把本说明扩展到后来新增的内容。

原文中的代码对应作者仓库的 MIT 许可；阅读页样式源自本地 `ultrascale-playbook-zh-tw`，其模板许可为 Apache 2.0，原项目维护者为 Twinkle AI Community。Markdown 渲染器 marked v12.0.2 使用 MIT 许可。各许可与来源记录保存在本项目的 `licenses/` 和 `NOTICE.md` 中。

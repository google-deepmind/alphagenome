![AlphaGenome header image](docs/source/_static/hero_poster.jpg)

# AlphaGenome API

![PyPI Python version](https://img.shields.io/pypi/pyversions/AlphaGenome)
![Presubmit Checks](https://github.com/google-deepmind/alphagenome/actions/workflows/presubmit_checks.yml/badge.svg)

[ [English](https://github.com/google-deepmind/alphagenome/blob/main/README.md) | 简体中文 ]

[**获取 API 密钥**](https://alphagenome.google/api) |
[**安装说明**](#安装) | [**快速上手**](#快速上手) |
[**官方文档**](https://www.alphagenomedocs.com/) |
[**社区论坛**](https://www.alphagenomecommunity.com) |
[**使用条款**](https://alphagenome.google/terms) |
[**模型代码**](https://github.com/google-deepmind/alphagenome_research)

`alphagenome` 软件包提供对 Google DeepMind 开发的 AlphaGenome 系列模型及 AlphaGenome Atlas（图谱）的编程访问接口：

-   **AlphaGenome 2**：我们最新的多模态 DNA 序列模型，具备更优的细胞类型特异性、对自有用户人类 scRNA-seq 数据的泛化能力（表达条件调节 / expression conditioning），以及针对 1 Mb 序列未观测模态的插补能力（imputation）。
-   **AlphaGenome Atlas**：覆盖 90 亿个人类突变的全基因组计算机模拟诱变（*in silico* mutagenesis），通过[零代码网站](https://alphagenome.google/atlas)与 API 提供预计算的突变效应评分以及 AlphaGenome 突变影响（AlphaGenome Variant Impact, AVI）评分。
-   **AlphaGenome 1**：[Avsec 等人 (2026)](https://www.nature.com/articles/s41586-025-10014-0) 中描述的初代模型。

该 API 面向[非商业用途](https://alphagenome.google/terms)免费提供。请求速率限制因需求而异：预计算的 Atlas 查询支持更高的请求速率，而实时模型预测则适用于小规模分析而非全基因组范围内的大规模分析。

获取支持与反馈：

-   **Bug 与代码问题**：
    [GitHub Issues](https://github.com/google-deepmind/alphagenome/issues)
-   **问题咨询与功能建议**：
    [社区论坛](https://www.alphagenomecommunity.com)
-   **其他咨询**：alphagenome@google.com

## 条款

AlphaGenome API 须遵守 [AlphaGenome 服务条款](https://alphagenome.google/terms)。除 AlphaGenome 服务条款中明确规定的情况（参见面向商业与非商业用途的许可使用可下载制品相关说明）外，AlphaGenome 生成的输出以及 AlphaGenome Atlas 中提供的其他信息仅限非商业用途，且不得用于训练其他机器学习模型。预测结果仅供理论建模与科学研究使用。严禁将其用于临床决策，亦不得作为医疗或其他专业建议的依据。

AlphaGenome 亦可通过 Google Cloud [用于商业用途](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/alphagenome)。

## 安装

从 [PyPI](https://pypi.org/project/alphagenome) 安装 `alphagenome`：

```bash
pip install alphagenome
```

有关搭建虚拟环境（`venv`、`uv` 或 `conda`）、源码安装以及配置 [AlphaGenome API 密钥](https://alphagenome.google/api)的详细说明，请参阅[安装指南](https://www.alphagenomedocs.com/installation.html)。

## 快速上手

请参阅[快速入门](https://www.alphagenomedocs.com/getting_started.html)以选择适合您任务的工具，并熟悉 AlphaGenome 2 与 AlphaGenome Atlas 的 API 工作流。

查阅[官方文档](https://www.alphagenomedocs.com/)获取交互式 Notebook 与指南：

-   [**教程 (Tutorials)**](https://www.alphagenomedocs.com/tutorials/index.html)：分步 Colab Notebook，涵盖参考条件设置、生成预测、突变效应评分、基于自有 scRNA-seq 数据的表达条件调节、绘图以及 AlphaGenome Atlas。
-   [**用户指南 (User guide)**](https://www.alphagenomedocs.com/user_guides/index.html)：关于数据对象、模型元数据、插补（imputation）、突变评分、可视化以及从 AlphaGenome 1 迁移的概念指南。
-   [**专题案例 (Vignettes)**](https://www.alphagenomedocs.com/vignettes/index.html)：端到端生物学分析与应用案例。
-   [**API 参考 (API reference)**](https://www.alphagenomedocs.com/api/index.html)：完整的 Python 包 API 参考。

例如，使用 AlphaGenome 2 对突变进行评分：

```python
from alphagenome.data import genome
from alphagenome.models.v2 import dna_client

dna_model = dna_client.create('YOUR_API_KEY')

# 选择实验条件：血液中的 CD14 阳性单核细胞。
condition = 'UBERON:0000178_CL:0001054'

# 选择基因组区域：以该突变为中心构建 1 Mb 区间。
variant = genome.Variant(
    chromosome='chr2',
    position=219243701,
    reference_bases='C',
    alternate_bases='T',
)
interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

# 使用所有推荐的评分器对突变进行评分，并格式化为 DataFrame。
scores = dna_model.score_variant(
    interval=interval,
    variant=variant,
    condition=condition,
)
```

## 贡献指南

我们欢迎为此项目提交问题修复以及部分新特性。在提交 Pull Request 之前，请阅读 [CONTRIBUTING](CONTRIBUTING.md) 文档，确保所有测试均通过，且贡献代码格式正确。

您可以在 `alphagenome` 根目录下运行以下命令来检查您的更改：

```shell
$ python3 -m pip install hatch
$ python3 -m hatch run check:all  # 运行 linter 与格式化检查。
$ python3 -m hatch test
```

## 引用 `alphagenome`

如果您在研究中使用了 AlphaGenome，请引用：

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

```bibtex
@article{alphagenome,
  title={Advancing regulatory variant effect prediction with {AlphaGenome}},
  author={Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drot{\'a}r, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
  journal={Nature},
  volume={649},
  number={8099},
  pages={1206--1218},
  year={2026},
  doi={10.1038/s41586-025-10014-0},
  publisher={Nature Publishing Group UK London}
}
```

如果您使用了 AlphaGenome Atlas 网站、API 或评分，请引用：

```bibtex
@article{alphagenome_atlas,
  title={{AlphaGenome} Atlas: in silico mutagenesis of the entire human genome improves prioritization and interpretation of non-coding variants},
  author={Cheng, Jun and Taylor, Kyle R., Nicolaisen, Lauren and Pan, Joshua and Bycroft, Clare and Perino, Matteo and Ward, Tom and Hawkes, Gareth and Covil, Laura E. and Weilert, Melanie and Thomas, Raina W. and Latysheva, Natasha and Hirschmann, J. Maile and Chen, Xi Dawn and Beaumont, Robin N. and Chundru, V Kartik and Weedon, Michael N. and Bourdareau, Simon and Chu, Hoyin and Hariharan, Dhavanthi and Kagohara, Thais and Ten\'{o}rio, Lucas and Ushigome, Yosuke and Shearer, Courtney A. and Ikica, Barbara and Fang, Ada and Naciri, Mouad and Johnston, Victoria and Green, Richard and Wong, Lai Hong and Dutordoir, Vincent and Mottram, Anne and Gayoso, Adam and Arvaniti, Eirini and Rehm, Heidi L. and Chen, Fei and Lareau, Caleb A. and Wright, Caroline F. and O'Donnell-Luria, Anne and Zeitlinger, Julia and Kohli, Pushmeet, and Avsec, {\v Z}iga },
  year = {2026},
  doi = {10.64898/2026.09.16.26363192},
  publisher = {Cold Spring Harbor Laboratory Press},
  journal = {medRxiv}
}
```

<!-- enableFinding(SNIPPET_INVALID_LANGUAGE) -->

## 致谢

AlphaGenome 与以下独立库和软件包进行交互通信和/或对其进行了引用：

*   [Abseil](https://github.com/abseil/abseil-py)
*   [anndata](https://github.com/scverse/anndata)
*   [fsspec](https://github.com/fsspec/filesystem_spec)
*   [gRPC](https://github.com/grpc/grpc)
*   [immutabledict](https://github.com/corenting/immutabledict)
*   [intervaltree](https://github.com/chaimleib/intervaltree)
*   [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
*   [matplotlib](https://matplotlib.org/)
*   [ml_dtypes](https://github.com/jax-ml/ml_dtypes)
*   [NumPy](https://numpy.org/)
*   [pandas](https://pandas.pydata.org/)
*   [protobuf](https://developers.google.com/protocol-buffers/)
*   [pyarrow](https://arrow.apache.org/)
*   [pyfaidx](https://github.com/mdshw5/pyfaidx/)
*   [SciPy](https://scipy.org/)
*   [seaborn](https://seaborn.pydata.org/)
*   [tqdm](https://github.com/tqdm/tqdm)
*   [typeguard](https://github.com/agronholm/typeguard)
*   [typing_extensions](https://github.com/python/typing_extensions)
*   [zstandard](https://github.com/indygreg/python-zstandard)

我们向所有这些库的贡献者与维护者表示由衷的感谢！

## 许可协议与免责声明

Copyright 2024 Google LLC

本仓库中的所有软件均基于 Apache License 2.0 (Apache 2.0) 获得许可；除非遵守 Apache 2.0 许可证，否则您不得使用本软件。您可以在以下地址获取 Apache 2.0 许可证副本：
https://www.apache.org/licenses/LICENSE-2.0。为免生疑问，如上所述，该 API 在遵守[使用条款](https://alphagenome.google/terms)的前提下面向非商业用途免费提供。

帮助您使用 AlphaGenome API 的示例与文档基于知识共享署名 4.0 国际许可协议 (Creative Commons Attribution 4.0 International License, CC-BY) 获得许可。您可以在以下地址获取 CC-BY 许可证副本：
https://creativecommons.org/licenses/by/4.0/legalcode。

除非适用法律要求或书面同意，依据 Apache 2.0 或 CC-BY 许可证分发的软件及材料均按“原样 (AS IS)”提供，不附带任何明示或暗示的担保或条件。有关许可证下权限和限制的具体语言，请参阅相关许可证。

这不是 Google 的官方产品。

### 第三方软件

您使用本仓库材料中引用的任何第三方软件、库或代码（包括[致谢](#致谢)部分列出的库）可能受单独的条款和条件或许可条款管辖。您使用第三方软件、库或代码须遵守任何此类条款，且您在使用前应确保能够遵守任何适用的限制或条款和条件。

### 参考数据集

为便于演示说明，随客户端代码包一起发布了 GENCODE 数据集的修改版本（可在此处获取：https://www.gencodegenes.org/human/releases.html），并参考以下内容提供：

-   Copyright © 2024 EMBL-EBI
-   GENCODE 数据集受 EMBL-EBI 使用条款约束，详见：https://www.ebi.ac.uk/about/terms-of-use。
-   引用文献：Frankish A, et al (2018) GENCODE reference annotation for the human and mouse genome.
-   有关 GENCODE 的更多详情可访问：https://www.gencodegenes.org/human/releases.html，其他引用信息可访问：https://www.gencodegenes.org/pages/publications.html，更多致谢内容可访问：https://www.gencodegenes.org/pages/gencode.html。

***

> **说明**：本文档为英文原版官方文档的中文翻译。如翻译版本与[英文原版](https://github.com/google-deepmind/alphagenome/blob/main/README.md)存在任何差异或不一致，请以英文原版为准。

# 模型与实际权重清单

本仓库提供 19 个配置、53 份已训练模型包。八份 Reference 权重同时包含在源码和 pip 安装包中。
所有 ZIP 均镜像到本仓库 v0.1.3 Releases，内容哈希与原项目 v0.1.0 一致。

## 配置总表

| 模型名 | 输入/视图 | EQR 目录接口 | 种子 | 权重位置 | 单个 ZIP 大小 MiB |
|---|---|---|---|---|---|
| `li2021_cnn` | 波形；单频 AG3 | 支持 | 20260928、20260929、20260930 | Releases，按需下载 | 1.88–1.88 |
| `gan2021_cnn` | 波形；单频 AG3 | 支持 | 20260928、20260929、20260930 | Releases，按需下载 | 2.88–2.89 |
| `gong_cnn` | 波形；单频 AG3 | 支持 | 20260928、20260929、20260930 | Releases，按需下载 | 0.05–0.05 |
| `gong_cnn_bilstm` | 波形；单频 AG3 | 支持 | 20260928、20260929、20260930 | Releases，按需下载 | 0.05–0.05 |
| `gan_cnn` | 波形；单频 AG3 | 支持 | 20260928、20260929、20260930 | Releases，按需下载 | 0.54–0.54 |
| `deeprfqc` | 波形；单频 AG3 | 支持 | 20260928、20260929、20260930 | Releases，按需下载 | 5.63–5.63 |
| `hegaz_capsule` | 波形；单频 AG3 | 支持 | 20260928、20260929、20260930 | Releases，按需下载 | 3.65–3.65 |
| `chen2026_image` | 波形；单频 AG3 | 支持 | 20260928、20260929、20260930 | Releases，按需下载 | 203.76–203.82 |
| `xiong2025_fcm` | 波形；单频 AG3 | 支持；完整台站集合 | 20260928、20260929、20260930 | Releases，按需下载 | 0.00–0.00 |
| `reference_ag3` | 波形；单频 AG3 | 支持 | 20260928、20260929、20260930 | 随包内置 + Releases | 1.40–1.40 |
| `reference_multifilter` | 波形；多频（1–7 视图） | 支持 | 20260928、20260929、20260930 | 随包内置 + Releases | 1.40–1.40 |
| `descriptors_ag3` | 六项描述量；单频 AG3 | 使用数组 API | 20260928、20260929、20260930 | Releases，按需下载 | 0.12–0.12 |
| `descriptors_multifilter` | 六项描述量；多频（1–7 视图） | 使用数组 API | 20260928、20260929、20260930 | Releases，按需下载 | 0.12–0.12 |
| `combined_ag3` | 波形 + 六项描述量；单频 AG3 | 使用数组 API | 20260928、20260929、20260930 | Releases，按需下载 | 1.41–1.41 |
| `combined_multifilter` | 波形 + 六项描述量；多频（1–7 视图） | 使用数组 API | 20260928、20260929、20260930 | Releases，按需下载 | 1.41–1.41 |
| `logreg_ag3` | 六项描述量；单频 AG3 | 使用数组 API | 20260928、20260929、20260930 | Releases，按需下载 | 0.00–0.00 |
| `logreg_multifilter` | 六项描述量；多频（1–7 视图） | 使用数组 API | 20260928、20260929、20260930 | Releases，按需下载 | 0.00–0.00 |
| `reference_ag1` | 波形；单频 AG1 | 支持 | 20260929 | 随包内置 + Releases | 1.41 |
| `reference_ag5` | 波形；单频 AG5 | 支持 | 20260929 | 随包内置 + Releases | 1.40 |

大小为 ZIP 压缩后文件大小，不是运行显存。Chen 图像模型较大，采用 Release 附件分发，不放进普通 Git 对象或默认 wheel。
不自动下载全部模型；默认只加载内置 Reference 多频模型。

## 内置模型实际位置

```text
src/rfqc_bench/pretrained/
  reference_ag1/seed20260929/
  reference_ag3/seed20260928/
  reference_ag3/seed20260929/
  reference_ag3/seed20260930/
  reference_ag5/seed20260929/
  reference_multifilter/seed20260928/
  reference_multifilter/seed20260929/
  reference_multifilter/seed20260930/
```

每个目录包含：

- `weights.safetensors`：实际神经网络参数。
- `bundle.json`：模型名、种子、训练得到的预处理参数、验证集固定阈值及权重哈希。

这些文件与 `.py` 网络定义一起进入 wheel，安装后可直接离线使用：

```python
from rfqc_bench import RFQCPredictor
ag3 = RFQCPredictor.from_pretrained("reference_ag3", seed=20260928)
multi = RFQCPredictor.from_pretrained("reference_multifilter", seed=20260928)
```

`from_pretrained()` 不需要提供 `model_dir`。若手动拷贝模型，必须同时拷贝 JSON 和 safetensors，不能只拷权重；
随后使用 `RFQCPredictor.from_directory("/path/to/bundle")` 或 CLI `--model-dir`。
FCM/LogReg 的已拟合统计参数存放在 JSON 中，没有神经权重文件。

## 所有 53 份模型的直接下载

下面链接都是当前社区仓库的 Release 附件。默认种子为各配置最早登记的种子，不依据测试精度挑选。
可用 `rfqc-bench download --model 模型名 --seed 种子` 下载并校验；内置模型则直接返回安装位置。

| 模型 | 种子 | ZIP |
|---|---|---|
| `chen2026_image` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/chen2026_image-seed20260928.zip) |
| `chen2026_image` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/chen2026_image-seed20260929.zip) |
| `chen2026_image` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/chen2026_image-seed20260930.zip) |
| `combined_ag3` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/combined_ag3-seed20260928.zip) |
| `combined_ag3` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/combined_ag3-seed20260929.zip) |
| `combined_ag3` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/combined_ag3-seed20260930.zip) |
| `combined_multifilter` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/combined_multifilter-seed20260928.zip) |
| `combined_multifilter` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/combined_multifilter-seed20260929.zip) |
| `combined_multifilter` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/combined_multifilter-seed20260930.zip) |
| `deeprfqc` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/deeprfqc-seed20260928.zip) |
| `deeprfqc` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/deeprfqc-seed20260929.zip) |
| `deeprfqc` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/deeprfqc-seed20260930.zip) |
| `descriptors_ag3` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/descriptors_ag3-seed20260928.zip) |
| `descriptors_ag3` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/descriptors_ag3-seed20260929.zip) |
| `descriptors_ag3` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/descriptors_ag3-seed20260930.zip) |
| `descriptors_multifilter` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/descriptors_multifilter-seed20260928.zip) |
| `descriptors_multifilter` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/descriptors_multifilter-seed20260929.zip) |
| `descriptors_multifilter` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/descriptors_multifilter-seed20260930.zip) |
| `gan2021_cnn` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gan2021_cnn-seed20260928.zip) |
| `gan2021_cnn` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gan2021_cnn-seed20260929.zip) |
| `gan2021_cnn` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gan2021_cnn-seed20260930.zip) |
| `gan_cnn` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gan_cnn-seed20260928.zip) |
| `gan_cnn` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gan_cnn-seed20260929.zip) |
| `gan_cnn` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gan_cnn-seed20260930.zip) |
| `gong_cnn` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gong_cnn-seed20260928.zip) |
| `gong_cnn` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gong_cnn-seed20260929.zip) |
| `gong_cnn` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gong_cnn-seed20260930.zip) |
| `gong_cnn_bilstm` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gong_cnn_bilstm-seed20260928.zip) |
| `gong_cnn_bilstm` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gong_cnn_bilstm-seed20260929.zip) |
| `gong_cnn_bilstm` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/gong_cnn_bilstm-seed20260930.zip) |
| `hegaz_capsule` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/hegaz_capsule-seed20260928.zip) |
| `hegaz_capsule` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/hegaz_capsule-seed20260929.zip) |
| `hegaz_capsule` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/hegaz_capsule-seed20260930.zip) |
| `li2021_cnn` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/li2021_cnn-seed20260928.zip) |
| `li2021_cnn` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/li2021_cnn-seed20260929.zip) |
| `li2021_cnn` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/li2021_cnn-seed20260930.zip) |
| `logreg_ag3` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/logreg_ag3-seed20260928.zip) |
| `logreg_ag3` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/logreg_ag3-seed20260929.zip) |
| `logreg_ag3` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/logreg_ag3-seed20260930.zip) |
| `logreg_multifilter` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/logreg_multifilter-seed20260928.zip) |
| `logreg_multifilter` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/logreg_multifilter-seed20260929.zip) |
| `logreg_multifilter` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/logreg_multifilter-seed20260930.zip) |
| `reference_ag1` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/reference_ag1-seed20260929.zip) |
| `reference_ag3` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/reference_ag3-seed20260928.zip) |
| `reference_ag3` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/reference_ag3-seed20260929.zip) |
| `reference_ag3` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/reference_ag3-seed20260930.zip) |
| `reference_ag5` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/reference_ag5-seed20260929.zip) |
| `reference_multifilter` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/reference_multifilter-seed20260928.zip) |
| `reference_multifilter` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/reference_multifilter-seed20260929.zip) |
| `reference_multifilter` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/reference_multifilter-seed20260930.zip) |
| `xiong2025_fcm` | 20260928 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/xiong2025_fcm-seed20260928.zip) |
| `xiong2025_fcm` | 20260929 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/xiong2025_fcm-seed20260929.zip) |
| `xiong2025_fcm` | 20260930 | [下载](https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/xiong2025_fcm-seed20260930.zip) |

## 完整性与来源

机器可读的大小、SHA256 和内部文件哈希在 [`model_zoo.json`](../src/rfqc_bench/model_zoo.json)。
可运行以下命令核对源码内置权重；已有全部 ZIP 时增加 `--archives /path/to/zips`：

```bash
python scripts/verify_community_models.py
python scripts/verify_community_models.py --archives /path/to/zips
```

53 份模型是本项目的文献方法改编与 Reference/描述量对照，在原 RF 任务上训练；
不表示已经复现每位原作者的数据流程和原文分数，不包含相位拾取迁移权重。
因数据问题暂停的新 DB/YP 重训模型不在本发行中。对外部数据必须独立验证。
许可与来源见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

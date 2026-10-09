# 给出 EQR 目录，自动筛选并生成 record

本说明适用于社区版 v0.1.4（目录功能自原项目 v0.1.2 提供）。适合已经计算完成、以直达 P 波为零时刻的径向接收函数
（二进制 SAC `.eqr` 文件）。输入目录后自动读取波形、匹配不同滤波结果、调用
现有模型，输出保留文件清单。**不移动、删除或改写任何原始 EQR 文件。**

## 1. 安装和最简单的调用

需要 Python 3.10 或以上，推荐虚拟环境：

```bash
python -m pip install --upgrade "rfqc-bench[api] @ git+https://github.com/cangyeone/dnn-rfqc-community.git@v0.1.4"
rfqc-bench screen-eqr "/path/to/all_eqr"
```

不安装 HTTP 服务时，可以用不依赖 Git 的 wheel：

```bash
python -m pip install --upgrade https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.4/rfqc_bench-0.1.4-py3-none-any.whl
```

运行结束后，输入目录内生成 `record`。默认使用现有
`reference_multifilter` 模型、最早登记的种子及模型保存的验证集阈值。
Reference 的 AG1/AG3/AG5 单频与多频共 8 份权重随程序提供，默认模型无需下载，
安装好依赖后可离线使用。其余模型首次使用时从本仓库 Releases 下载对应权重并缓存，
不下载训练数据。全部 53 份模型参数与原项目 v0.1.0 权重一致，本次未加入仍在进行的
新 DB/YP 重训权重。完整清单见 [MODELS_zh.md](MODELS_zh.md)。本地模型见第 4 节。

Windows 示例：

```powershell
rfqc-bench screen-eqr "D:\my_eqr"
```

WSL 对应路径为 `/mnt/d/my_eqr`。输入必须是**已经解压的目录**；本接口不自动
解压 `.tar`、`.rar`，也不读取仅包含波形数字的 `good_xxx` 导出文本。

Linux、WSL、macOS 用户可在自己的目录独立安装：

```bash
git clone https://github.com/cangyeone/dnn-rfqc-community.git
cd dnn-rfqc-community
bash install.sh
bash screen_eqr.sh "/path/to/eqr" --threshold 0.7 --resample --output "$PWD/results/record"
```

`install.sh` 使用 `python3`，也可用 `RFQC_PYTHON=/path/to/python bash install.sh`
指定 Python 3.10+；只写入本目录的 `.venv`，无需 sudo。已有环境可以通过
`RFQC_PYTHON=/path/to/venv/bin/python bash screen_eqr.sh ...` 使用。
模型缓存默认在各用户自己的 `~/.cache/rfqc-bench`，可用 `--cache-dir` 指定。
输入只读时用 `--output` 指向自己可写的位置。脚本没有固定用户名或开发者路径。
虚拟环境不要跨电脑复制；移动源码后在新位置重新运行安装。
通过该脚本安装后，直接执行 `bash screen_eqr.sh` 即可；使用 `rfqc-bench` 命令或
Python API 前先执行 `source .venv/bin/activate`，或显式使用 `.venv/bin/python`。

纯 CPU 安装可执行 `RFQC_TORCH_INDEX=https://download.pytorch.org/whl/cpu bash install.sh`。
CUDA 用户应按 GPU/驱动选择 PyTorch 轮子源，或先在自己的环境安装合适的 PyTorch
再用 pip 安装本包。筛选默认 CPU。

### 阈值和重采样

`--threshold 0.7` 设置保留规则为 `p_good >= 0.7`。阈值范围为 0～1（含边界），
0 保留全部有效候选，1 仅保留分数恰为 1 的候选；无效文件始终进入异常清单。
提高阈值会减少或保持保留数量，不保证精度一定提高。省略时使用模型的验证集阈值。
覆盖仅对本次调用生效，不改模型、默认阈值或其他 HTTP 请求。

`--resample` 从每个 SAC 的 delta 读取采样率，例如 20 Hz、50 Hz、5 Hz，转换为
模型要求的 10 Hz（0.1 s、501 点、−10～40 s）。同一目录、同一事件可混合不同采样率。
不开启时保持旧版严格检查。处理只发生在内存中；`record` 仍指向原始 `.eqr`，
不另外导出重采样 SAC。

## 2. 目录、文件名和频率匹配

推荐保留下面的原始目录结构，可以一次传入多个台站的上层目录：

```text
all_eqr/
  DB_EW27/
    AG1/EW27_20161107_2131.eqr
    AG3/EW27_20161107_2131.eqr
    AG5/EW27_20161107_2131.eqr
  YP_NE8A/
    AG3/NE8A_10129_060616.eqr
```

程序递归扫描，以 AG 目录的上级目录区分台站，以同一台站内**完全相同的文件名**
匹配事件，再按 Gaussian 系数排列输入。AG1、AG1.5、AG2、AG2.5、AG3、AG4、AG5
是 Gaussian 系数，**不是 Hz**。多频模型允许每条记录只有 1–7 个视图，缺 AG5
无需补造文件。输入目录自身是 `AG3` 等目录时也能识别。

可明确只用部分滤波视图：

```bash
rfqc-bench screen-eqr "/path/to/all_eqr" --filters 1 1.5 3 5 --resample
rfqc-bench screen-eqr "/path/to/all_eqr" --model reference_ag1 --filters 1 --output "record_ag1"
rfqc-bench screen-eqr "/path/to/all_eqr" --model reference_ag5 --filters 5 --output "record_ag5"
```

省略 `--filters` 时使用模型所需的可用视图。单频模型须与实际 AG 系数匹配，不能把
AG5 当 AG3 输入；多频模型可以使用支持集合中任意非空的可用子集。当前权重仅支持
1、1.5、2、2.5、3、4、5，不会将未训练的 AG6 或任意 Hz 带通标签冒充其中之一。
重采样不会将 AG1 变为 AG3，也不生成缺失滤波视图。

`bad/`、`good/` 等子目录名不作为人工标签或预测依据。相同台站、同一事件、同一
AG 若重复出现，整条事件被列入异常清单，不能用文件所在的 good/bad 目录代替预测。
跨频率已提供的台站名和事件几何头段若相互冲突，也会拒绝该事件。

如果所有文件都直接放在一个目录，且路径中没有 AG 标识，必须明确其滤波系数：

```bash
rfqc-bench screen-eqr "/path/to/flat_eqr" --gaussian 3
```

不能从波形自动猜测滤波系数。平铺目录按 SAC 台站/台网头段区分台站；文件名仍需
唯一标识事件。不同数据版本不要混在同一台站目录下。

## 3. record 的格式

`record` 是 UTF-8 文本，无表头，每行一个相对于输入目录的文件路径。例如：

```text
DB_EW27/AG1/EW27_20161107_2131.eqr
DB_EW27/AG3/EW27_20161107_2131.eqr
DB_EW27/AG5/EW27_20161107_2131.eqr
```

多频模型输出的是**事件的联合判断**：预测 good 后，该事件实际参与判断的所有
有效视图都写入 `record`；这不等于每个频率都独立通过一次分类。单频模型只输出
它实际使用的那个频率文件。路径保留子目录，避免多个台站或频率中的同名文件混淆。
输入为平铺目录时，每行自然就是 `.eqr` 文件名。

同时生成：

| 文件 | 内容 |
|---|---|
| `record.predictions.csv` | 每个有效事件的 good 分数、good/bad 判断、本次阈值、模型、种子、文件路径和来源 SHA256 |
| `record.rejected.csv` | 无法读取、无有效时间窗、重复/冲突事件、单频模型未使用的视图及原因 |
| `record.inputs.csv` | 参与判断文件的原始样点数、delta/Hz、b/e、是否重采样、比例、处理方法、SHA256 |
| `record.json` | 数量、模型原阈值/本次阈值及来源、视图选择、目标网格、重采样文件数、输出路径和哈希 |

有效输入全部预测 bad 时，`record` 是正常的空文件；没有任何有效事件时，报告为
`no_valid_events`，命令行返回退出码 2。应检查异常清单，不能把读取失败解释为预测 bad。
存在损坏视图的多频事件可使用剩余有效视图判断，损坏文件不会写入保留清单。
符号链接不读取，扫描跳过的链接目录也会列入异常清单。

默认不覆盖已有输出，重复运行会提示选择新文件；需要重做时可指定新输出：

```bash
rfqc-bench screen-eqr "/path/to/all_eqr" --output "/path/to/results/record_multifilter"
```

明确替换旧输出时使用 `--overwrite`。发生读取后的推理故障或中断，不会发布半份
保留名单；原有结果保持不变。请勿在筛选过程中修改输入文件。

## 4. 换模型、用 GPU 或新训练权重

```bash
rfqc-bench screen-eqr "/path/to/all_eqr" --model reference_ag3 --output "/path/to/record_ag3"
rfqc-bench screen-eqr "/path/to/all_eqr" --model gong_cnn_bilstm --device cuda:0 --output "/path/to/record_gong"
rfqc-bench screen-eqr "/path/to/all_eqr" --model-dir "/path/to/completed_run" --output "/path/to/record_new"
```

本地模型目录应包含 `bundle.json` 和 `weights.safetensors`（FCM 不需要神经权重）。
预处理随模型加载，不在待筛选数据上重新拟合；阈值默认随模型，也可显式覆盖。默认 CPU；可改为 `cuda:0`、
`cuda:1` 或 macOS 的 `mps`。`--batch-size 32` 控制神经模型推理批量。

目录接口支持现有纯波形模型：Li-CNN、Gan-CNN、Gong-CNN/BiLSTM、DeepRFQC、
RF-Capsule、Chen-AlexNet、Xiong-FCM，以及 Reference 的单频/多频配置。
需要六个外部描述量的 descriptor/combined/LogReg 模型仍需用原来的数组 API。
接口不会从 SAC 头段编造这些描述量。

Xiong-FCM 的站内相关性依赖输入集合。程序按台站整组推理，不按神经网络批量切开
该集合。请给它该台站的完整候选集合；只给子集会改变结果。

## 5. Python 接口

```python
from rfqc_bench import screen_eqr, RFQCPredictor

report = screen_eqr("/path/to/all_eqr")
print(report["good_events"], report["retained_files"], report["outputs"]["record"])

# 多次调用时只加载一次模型；输出路径相对于 Python 当前工作目录。
model = RFQCPredictor.from_pretrained("reference_multifilter", device="cpu")
report = model.screen_eqr("/path/to/another_station", output="results/record",
                          threshold=0.7, filters=[1, 3, 5], resample=True)
```

路径含空格或中文都可以。预测分数表示与所用人工标签体系的一致性，不是经过校准的
地球物理可用性概率。新区域数据建议结合波形检查，不保证所有自动保留 RF 都可用于反演。

## 6. HTTP 接口

安装 `[api]` 扩展后，启动时指定可读取及写入的服务器目录：

```bash
rfqc-bench serve --model reference_multifilter --eqr-root "/srv/rfqc" --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/screen-eqr \
  -H "Content-Type: application/json" \
  -d '{"directory":"incoming/all_eqr", "output":"results/record", "threshold":0.7, "filters":[1,3,5], "resample":true}'
```

上述路径均相对于服务器 `/srv/rfqc`，不是发请求的客户端目录。文件需先放在服务器上。
响应包含输出文件路径和筛选统计；Swagger 文档在 `/docs`。平铺目录可在请求中增加
`"gaussian":3`。未配置 `--eqr-root` 时此端点返回 403；路径越界/符号链接拒绝；
已有输出返回 409，不通过 HTTP 覆盖。目录扫描默认最多 100000 个 EQR 文件，
可用 `--max-eqr-files` 调整。请求同步执行，大数据推荐 CLI。

服务默认只绑定本机；现有 API 不提供身份认证，不应直接开放到公网。现有
`POST /predict` 数组接口保持兼容。

## 输入时间轴要求

支持二进制 SAC v6/v7 大小端、等间隔时间序列。输入必须实际覆盖以 P=0 为基准的
−10 到 40 秒。常见 `b=-15, e=45, npts=601` 自动截取 501 点，即使开启重采样也不改样点。
其他网格开启 `--resample` 后使用
[SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
的零相位多相 FIR（Kaiser 5、线性边界延拓），降采样包含抗混叠滤波，再线性对齐目标
时间点。边界延拓仅用于数值滤波，输入本身必须覆盖完整时间窗，不补零补造缺失数据。
上采样不能恢复原本缺少的高频信息；不同采样率/区域的模型适用性仍需检查。
程序不寻找 P 波、不改变零点，也不将原始地震记录直接当接收函数。

库原有预处理仍由所选模型执行，SAC 读取层保留原始幅度。不使用 baz/gcarc/user4
作为质量标签；读取这些头段仅用于检查跨滤波视图是否对应一致。

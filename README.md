# DNN RFQC Community

接收函数自动质量筛选：给出 `.eqr` 文件目录，生成保留文件名单 `record`。
提供 **单频与多频模型、实际训练权重、命令行、Python API 和 HTTP API**。

v0.1.5 支持自定义阈值、选择滤波视图，以及将不同采样率的 SAC 重采样到模型网格。

**8 份 Reference 权重直接随源码和 pip 安装包提供**，无需再下载模型；全部
**19 个配置、53 份已训练模型包**同时放在[本仓库 Releases](https://github.com/cangyeone/dnn-rfqc-community/releases/tag/v0.1.3)。
不包含观测波形、人工标签或逐条实验预测。

[详细目录筛选说明](docs/EQR_SCREENING_zh.md) · [模型与权重清单](docs/MODELS_zh.md) ·
[Python API](docs/API.md) · [HTTP 部署](docs/DEPLOYMENT.md) ·
[输入格式](docs/DATA.md) · [训练与续训](docs/TRAINING.md) ·
[方法来源](docs/METHODS.md) · [对比与测速](docs/BENCHMARK_zh.md)

## 1. 安装

需要 Python 3.10 或以上，推荐单独的虚拟环境。可直接通过 pip 安装 wheel，无需 Git：

```bash
python -m pip install https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.1.5/rfqc_bench-0.1.5-py3-none-any.whl
rfqc-bench doctor
```

或者克隆源码安装；单频与多频权重也在源码中：

```bash
git clone https://github.com/cangyeone/dnn-rfqc-community.git
cd dnn-rfqc-community
bash install.sh
bash screen_eqr.sh "/path/to/all_eqr" --threshold 0.7 --resample
```

通过 `install.sh` 安装后，`bash screen_eqr.sh` 无需激活环境；若要直接运行
`rfqc-bench` 或 `python screen_eqr.py`，先执行 `source .venv/bin/activate`。

HTTP 服务需要额外依赖，在仓库目录运行 `python -m pip install -e ".[api]"`，
或直接安装指定版本：

```bash
python -m pip install "rfqc-bench[api] @ git+https://github.com/cangyeone/dnn-rfqc-community.git@v0.1.5"
```

为兼容原接口，**pip 包名仍为 `rfqc-bench`，Python 导入名为 `rfqc_bench`**。
`dnn-rfqc` 与 `rfqc-bench` 两个命令等价。以上通过 GitHub 安装，不宣称已经发布到
PyPI。已有旧版可加 `--upgrade`，建议使用独立环境，避免与其他源码安装混淆。
CUDA 机器先安装适配自身硬件/驱动的 PyTorch，再安装本包；默认使用 CPU。

## 2. 单频和多频各一条命令

```bash
# 多频联合判断：默认 Reference 多频模型
rfqc-bench screen-eqr "/path/to/all_eqr"

# 单频 AG3：只使用 AG3，输出也只列出 AG3 文件
rfqc-bench screen-eqr "/path/to/all_eqr" --model reference_ag3 --output "results/record_ag3"

# GPU 推理；也可以用 --device mps（支持的 macOS）
rfqc-bench screen-eqr "/path/to/all_eqr" --device cuda:0 --batch-size 32 --output "results/record_multi"
```

Windows 示例：`rfqc-bench screen-eqr "D:\my_eqr"`；WSL 对应路径为 `/mnt/d/my_eqr`。
源码安装后也可运行 `python screen_eqr.py "/path/to/all_eqr"`，参数与 `screen-eqr` 一致。

```bash
# 自定义阈值，按各 SAC 头段识别采样率并转换到模型网格
rfqc-bench screen-eqr "/path/to/all_eqr" --threshold 0.7 --resample --output "results/record_t07"
# 仅使用 AG1、AG3、AG5 的可用视图进行联合判断
rfqc-bench screen-eqr "/path/to/all_eqr" --filters 1 3 5 --resample --output "results/record_135"
```

保留规则为 `p_good >= threshold`；省略阈值时使用模型的验证集阈值。
`--filters` 是高斯滤波系数，`--resample` 处理时间采样率，两者不同。
`install.sh` 只创建当前目录的 `.venv`，没有固定用户名，无需 sudo。其他用户在自己的
电脑安装依赖即可，不需要开发者的 Python 环境；Windows 使用上述 pip 命令。

## 3. 数据怎么放

保留台站和 AG 目录，同一台站中跨滤波视图的同一事件使用**完全相同的文件名**：

```text
all_eqr/
  DB_EW27/
    AG1/event_001.eqr
    AG3/event_001.eqr
    AG5/event_001.eqr
    AG3/event_002.eqr
  YP_NE8A/
    AG1/event_003.eqr
    AG3/event_003.eqr
```

- 多频模型接受每个事件实际可用的 1–7 个视图，缺 AG5 无需补造文件。
- AG1、AG1.5、AG2、AG2.5、AG3、AG4、AG5 是 Gaussian 系数，**不是 Hz**。
- `good/`、`bad/` 子目录不作为预测依据。原始文件不移动、不删除、不改写。
- 输入应为已计算的径向 RF，二进制 SAC `.eqr`，直达 P 波位于 `t=0`，采样间隔
  默认要求 0.1 s，实际覆盖 −10 到 40 s；加 `--resample` 可转换其他采样率并对齐时间网格。
- 不能补造缺失时间窗，不会改变高斯系数或从原始地震记录计算 RF。压缩包应先解压；
  本目录接口不解析仅含数字波形的 `good_xxx` 导出文本。

平铺目录没有 AG 标识时必须明确实际滤波系数：

```bash
rfqc-bench screen-eqr "/path/to/flat_eqr" --gaussian 3
```

不能凭波形猜测频率，也不能把不同频率混放后用一个系数冒充。
重复文件、缺失视图、时间轴及异常处理详见[完整指南](docs/EQR_SCREENING_zh.md)。

## 4. 输出 record

默认在输入目录生成 `record`：UTF-8，无表头，每行一个去重后的 `.eqr` 文件名，
不含 `AG*/` 或台站等目录前缀。例如同一文件在 AG1、AG3、AG5 都被保留，只输出一行：

```text
XZ_NAQ_2022219_214001.eqr
```

多频 good 判断保留该事件实际参与联合判断的有效视图，不表示每个视图单独通过分类。
单频模型只使用其要求的视图。同名文件在整个输出中只出现一次；完整来源路径保留在 CSV 明细中。
`record.json` 的 `record_entries` 是名单行数，`retained_files` 是去重前保留的实际视图文件数。
不同模型建议用 `--output` 命名，不覆盖已有结果；
明确替换才使用 `--overwrite`。

| 文件 | 内容 |
|---|---|
| `record` | 保留 `.eqr` 文件名单 |
| `record.predictions.csv` | 事件分数、判断、阈值、模型、种子、来源路径和 SHA256 |
| `record.rejected.csv` | 无法处理或未使用文件及其原因 |
| `record.inputs.csv` | 原始采样率、时间范围、重采样方式与源文件哈希 |
| `record.json` | 数量统计、模型标识、输出位置与文件哈希 |

全部有效事件预测 bad 时名单为空；若没有有效输入，报告 `no_valid_events`，退出码为 2。
应检查异常清单，不能把读取失败当成预测 bad。

## 5. 模型确实包含在程序中

| 内置模型 | 输入 | 随包提供的种子 |
|---|---|---|
| `reference_ag1` | 单频 AG1 | 20260929 |
| `reference_ag3` | 单频 AG3 | 20260928、20260929、20260930 |
| `reference_ag5` | 单频 AG5 | 20260929 |
| `reference_multifilter` | 多频，可缺视图 | 20260928、20260929、20260930 |

实际权重在 [`src/rfqc_bench/pretrained/`](src/rfqc_bench/pretrained/)，每份包含
`bundle.json` 和 `weights.safetensors`，不是 Git LFS 指针或随机网络。
`from_pretrained()` 自动使用内置文件并核验哈希；默认选择最早登记种子，不按测试分数挑选。
安装好依赖后，上述模型可断网加载、预测。

Li-CNN、Gan-CNN、Gong-CNN/BiLSTM、DeepRFQC、RF-Capsule、Chen-AlexNet、Xiong-FCM、
描述量和 LogReg 等模型也已上传到**本仓库** Releases。首次指定时只下载对应一个模型，
缓存位置默认为 `~/.cache/rfqc-bench`，可用 `RFQC_CACHE` 修改。
全部配置、种子、大小、下载方式见[模型清单](docs/MODELS_zh.md)。

```bash
rfqc-bench screen-eqr "/path/to/all_eqr" --model gong_cnn_bilstm --output "record_gong"
rfqc-bench download --model deeprfqc --seed 20260928
rfqc-bench screen-eqr "/path/to/all_eqr" --model-dir "/path/to/local_bundle" --output "record_custom"
```

目录接口支持纯波形模型。描述量/combined/LogReg 仍需在数组 API 中提供六项描述量，
不能从 SAC 头段编造。Xiong-FCM 必须提供同台站完整候选集合，程序按站计算。
预处理参数和验证集阈值均随模型加载，不在新输入上重新拟合。目录接口可为单次调用
显式覆盖阈值，同时记录原阈值和本次阈值，不修改模型文件。

## 6. Python 调用

```python
from rfqc_bench import RFQCPredictor, screen_eqr

report = screen_eqr("/path/to/all_eqr", output="results/record_multi",
                    threshold=0.7, filters=[1, 3, 5], resample=True)
print(report["good_events"], report["retained_files"])

single = RFQCPredictor.from_pretrained("reference_ag3", device="cpu")
multi = RFQCPredictor.from_pretrained("reference_multifilter", device="cpu")
single.screen_eqr("/path/to/another_station", output="results/record_ag3")
multi.screen_eqr("/path/to/another_station", output="results/record_multi_2")
```

已有数组/NPZ 使用 `RFData` 与 `.predict()`，详见 [API](docs/API.md) 和
[输入格式](docs/DATA.md)。`create_model()` 只创建随机网络；筛选应使用加载好的 predictor。

## 7. HTTP 调用

```bash
rfqc-bench serve --model reference_multifilter --eqr-root "/srv/rfqc" --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/screen-eqr \
  -H "Content-Type: application/json" \
  -d '{"directory":"incoming/all_eqr", "output":"results/record_multi", "threshold":0.7, "resample":true, "filters":[1,3,5]}'
```

路径相对于服务器 `/srv/rfqc`，文件应先放在服务器；不是客户端电脑的路径。
交互文档：`http://127.0.0.1:8000/docs`，原 `POST /predict` 数组接口仍可使用。
单频服务将模型改为 `reference_ag3`；同时开两个服务应指定不同端口。
目录接口默认关闭，必须配置 `--eqr-root`；越界路径拒绝，已有输出返回 409。
默认仅本机访问，对外服务需要自行配置认证。详见[部署说明](docs/DEPLOYMENT.md)。

## 8. 验证、复现与适用范围

```bash
rfqc-bench doctor
rfqc-bench demo --model reference_multifilter --output synthetic_predictions.csv
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/verify_community_models.py
```

演示与单元测试使用合成波形，验证安装而不是科学精度。模型校验脚本不重新训练。
训练/续训方法见[训练说明](docs/TRAINING.md)，历史结果与推理测速见
[对比说明](docs/BENCHMARK_zh.md)。安装和预测不会启动训练队列、定时任务或开机服务。

社区版本基于 RFQC Bench `d44b016`，本次所有权重与原公开 v0.1.0 参数逐字节一致。
它们是本项目在 RF 数据上训练的文献方法改编版，不是各论文作者的原始预训练模型，
不包含相位拾取迁移权重。修正数据后的新 DB/YP 重训单独进行，尚未核验的权重**没有混入**本次发布。
good 分数不是校准后的物理可用性概率，新地区的适用性仍需独立评价。

软件与模型按 GPL-3.0-only 发行，来源与第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
和 [CITATION.cff](CITATION.cff)；软件许可不代表数据集许可。
问题反馈：[Issues](https://github.com/cangyeone/dnn-rfqc-community/issues)。

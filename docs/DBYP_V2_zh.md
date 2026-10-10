# v0.2.0：训练完成后的升级与使用说明

本版正式使用 2026-10-09 修正后的 DB/YP 数据（`dbyp_complete_20261009_v2`）重训权重。
10 个配置各 3 次，共 30 次已完成。原来的 EQR 目录接口、阈值覆盖、重采样和去重名单仍可使用。
升级会改变模型、预处理参数和默认阈值，因此重新筛选的名单可能与 v0.1.5 不同。

## 1. 安装或升级

在你自己的 Python 3.10+ 环境中运行，无需训练数据或开发者账号：

```bash
python -m pip install --upgrade https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.2.0/rfqc_bench-0.2.0-py3-none-any.whl
python -c "import rfqc_bench; print(rfqc_bench.__version__, rfqc_bench.__file__)"
rfqc-bench doctor
```

版本应显示 `0.2.0`。若显示旧版或另一份源码目录，请在实际运行筛选的那个环境安装。
本版通过 GitHub wheel/Git 安装，没有发布到 PyPI。

源码新安装：

```bash
git clone --branch v0.2.0 https://github.com/cangyeone/dnn-rfqc-community.git
cd dnn-rfqc-community
bash install.sh
bash screen_eqr.sh "/path/to/all_eqr" --resample
```

已有源码工作目录且没有本地改动时，可先 `git pull --ff-only`，再 `bash install.sh`。
安装脚本只创建本目录的 `.venv`。运行启动脚本无需激活；直接使用 Python API 或
`rfqc-bench` 命令前先 `source .venv/bin/activate`。Windows 使用上面的 pip 方式。
服务器上的虚拟环境不要复制给其他用户；每人在自己的电脑/目录安装即可。

## 2. 筛选新数据：单频、多频、其他方法

保留 `台站/AG系数/同名事件.eqr` 结构；Gaussian 系数不是采样率 Hz。
输入为直达 P 波对应 `t=0` 的径向 SAC RF，实际覆盖 −10 到 40 s。
`--resample` 依据 SAC 采样间隔转换到模型的 10 Hz 网格，不生成缺失时间段。

```bash
# 多频：默认固定种子 20260928，默认阈值 0.90
rfqc-bench screen-eqr "/path/to/all_eqr" --resample --output "results/record_multi"

# 单频 AG3：默认固定种子 20260928，默认阈值 0.86
rfqc-bench screen-eqr "/path/to/all_eqr" --model reference_ag3 --resample --output "results/record_ag3"

# 本轮 Gong-CNN-BiLSTM；首次使用自动下载该模型，不下载全部模型
rfqc-bench screen-eqr "/path/to/all_eqr" --model gong_cnn_bilstm --resample --output "results/record_gong"

# 指定 GPU、自定义阈值与参与联合判断的视图
rfqc-bench screen-eqr "/path/to/all_eqr" --device cuda:0 --threshold 0.8 --filters 1 3 5 --resample --output "results/record_custom"
```

省略 `--threshold` 才使用随模型保存的验证集阈值。保留规则为 `p_good >= threshold`；
更高阈值意味着更少或相同的保留量，不保证其他地区精度提高。本轮报告分数使用完整的已登记输入和各自冻结阈值，
不能直接套用于自定义阈值或删减视图后的结果。不同种子也可能具有不同阈值。

多频模型允许每条记录有 1–7 个可用视图，缺 AG5 不需补造；AG3 单频必须有 AG3。
单层平铺文件夹需要提供真实系数，如 `--model reference_ag3 --gaussian 3`。
`good/`、`bad/` 文件夹名不会被当成预测标签。不能混合不同系数后统一冒充 AG3。

Windows 输入示例：`"D:\my_eqr"`；在 WSL 中使用 `"/mnt/d/my_eqr"`。
现有项目服务器的新入口为：

```bash
bash /home/yuzy/rfqc_eqr_interface_v020/screen_eqr.sh "/mnt/d/你的eqr目录" --resample
```

原 `/home/yuzy/rfqc_eqr_interface_v012/screen_eqr.sh` 入口也指向新版。
以上两个路径是当前项目服务器的部署位置；其他用户使用自己的安装目录。

## 3. 输出文件与旧名单

`record` 无表头，只保留文件名，全局去重并去除 `AG*/`、台站等所有目录前缀，例如：

```text
XZ_NAQ_2022219_214001.eqr
```

相同事件的多个滤波视图被联合保留时，名单只写一行。完整来源路径、分数、判断和阈值在
`record.predictions.csv`；采样与重采样审计在 `record.inputs.csv`；无效/未使用输入在
`record.rejected.csv`。`record.json` 中 `record_entries` 是去重行数，`retained_files`
是保留的实际视图文件数。原 SAC 文件不会被改写、移动或删除。

已有输出不会自动转换或覆盖。建议先用 `--output` 保存新名单，与旧名单对比；明确要替换时加 `--overwrite`。
正在运行的 Python/HTTP 进程升级后需要重新加载模型或重启服务，才会使用新权重。

## 4. Python 与 HTTP

```python
from rfqc_bench import RFQCPredictor, available_weights

model = RFQCPredictor.from_pretrained("reference_multifilter", seed=20260928)
print(model.threshold)  # 0.90
print(model.bundle["weights_sha256"])
report = model.screen_eqr("/path/to/all_eqr", output="results/record", resample=True)
print(report["good_events"], report["record_entries"])

for item in available_weights():
    if item["name"] == "reference_multifilter":
        print(item["seed"], item["dataset_version"], item["release"])
```

HTTP 需要 API 依赖，先在目标环境执行：

```bash
python -m pip install --upgrade "rfqc-bench[api] @ git+https://github.com/cangyeone/dnn-rfqc-community.git@v0.2.0"
rfqc-bench serve --model reference_multifilter --eqr-root "/srv/rfqc" --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/screen-eqr -H "Content-Type: application/json" \
  -d '{"directory":"incoming/all_eqr","output":"results/record","resample":true,"threshold":0.8}'
```

API 的路径指向服务器文件，详细约定见 [HTTP 部署](DEPLOYMENT.md) 和 [Python API](API.md)。

## 5. 哪些模型已更新，训练结果如何

本轮更新：`li2021_cnn`、`gan2021_cnn`、`gong_cnn_bilstm`、`gan_cnn`、`deeprfqc`、
`hegaz_capsule`、`chen2026_image`、`reference_ag3`、`reference_multifilter`、`xiong2025_fcm`。
每个配置均提供 20260928、20260929、20260930 三个种子；默认固定使用第一个，不挑最高测试分数，也不自动集成。

主数据共 163,889 条、205 个台站。训练/验证/测试分别为 112,393 / 23,673 / 27,823 条，
对应 142 / 31 / 32 个互不重叠台站。全部方法从头训练或拟合，不使用相位任务迁移；
神经网络使用均衡采样、最多 50 轮、早停 10 轮、批量 32。预处理只拟合训练集；
验证集 good AP 选择检查点，验证集 Macro-F1 选择阈值。环境为 PyTorch 2.8.0+cu128、RTX 5090。

完整十方法表（各列最高均值加粗）见 [本轮精度结果](../benchmarks/2026-10-09-dbyp-v2/COMPARISON.md)。
AG3、多频、Gong-BiLSTM 的平均准确率分别为 95.43%、95.54%、96.06%。多频减 AG3 的
配对差值为 +0.105 ± 0.235 个百分点；增益较小，不能据此声称普遍优于单频或其他方法。
± 表示三个种子的样本标准差，不是新地区泛化置信区间。准确率也不等于保留名单的 good Precision。

小震级波形无人工标签，未加入训练或上述准确率计算。现有小震级筛选与叠加图是描述性结果；
缺少可匹配反方位角/射线参数的记录不生成对应物理量。主数据的 H–κ 已计算供人工审阅，
不能仅凭模型分数或 H–κ 接近便认定地壳参数准确。目录筛选命令本身不执行 H–κ。

`reference_ag1`、`reference_ag5`、`gong_cnn`、descriptors、combined、LogReg 共 23 份模型
是保留的历史权重，本轮未重训。每份来源都写入模型目录的 `dataset_version` 和 `release`，
完整下载链接见 [模型清单](MODELS_zh.md)。旧版软件和权重继续保留在历史 Release；
严格复现旧筛选时使用独立环境安装 v0.1.5，或用 `--model-dir` 指向明确的历史 bundle。

## 6. 核验与复算

```bash
python scripts/verify_community_models.py
python scripts/reproduce_dbyp_summary.py
python -m pytest -q
```

新权重与已完成训练输出逐字节一致；原始 30 份完成凭据及 207 个文件哈希已经核对，
相同测试 ID/标签和逐样本分类指标在实验主机复算通过。公开的三个 CSV 仅含种子汇总和配对差值，
可复算均值/标准差；没有上传波形、人工标签、样本名单或逐条预测。
来源协议、哈希与导出核验保存在 `benchmarks/2026-10-09-dbyp-v2/` 和
`validation/dbyp_v2_export_20261010.json`。这套权重是本项目的文献方法改编，不是原作者预训练权重。

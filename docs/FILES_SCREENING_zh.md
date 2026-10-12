# EQR 目录与 files 波形导出：选择输入格式

v0.2.1 增加 `--input-type eqr|files`，使用相同的已训练模型生成 `record`。
默认值为 `eqr`；原有命令省略该参数时仍按二进制 SAC `.eqr` 目录读取。
`files` 用于每行带文件名、采样信息和波形的文本导出，如 `YP_NE3A/AG3/good_1667`。
模型仍是 v0.2.0 已核验的 DB/YP v2 权重；本次没有重新训练。

## 安装或升级

```bash
python -m pip install --upgrade https://github.com/cangyeone/dnn-rfqc-community/releases/download/v0.2.1/rfqc_bench-0.2.1-py3-none-any.whl
python -c "import rfqc_bench; print(rfqc_bench.__version__)"
```

版本应显示 `0.2.1`。源码安装用户更新仓库后重新执行 `bash install.sh`。
现有 Python 或 HTTP 进程需重新加载模型/重启服务后才会使用新代码。

## 两种输入各一条命令

```bash
# 二进制 SAC EQR 目录：原有用法
rfqc-bench screen-eqr "/path/to/eqr目录" --input-type eqr --resample

# 记录波形的 files 目录：新增用法，先解压 ZIP
unzip YP_NE3A.zip -d inputs
rfqc-bench screen-eqr "inputs/YP_NE3A" --input-type files
```

源码启动脚本和项目服务器的兼容入口使用完全相同的参数：

```bash
bash screen_eqr.sh "/path/to/YP_NE3A" --input-type files
bash /home/yuzy/rfqc_eqr_interface_v012/screen_eqr.sh "/mnt/d/YP_NE3A" --input-type files
```

服务器命令中的目录是当前部署示例；其他用户使用自己的安装目录。
ZIP 必须先解压，不能把压缩包本身当作波形文本传入。

## files 的格式

保留台站/AG 结构；同一台站的同一事件在不同视图中使用相同文件名：

```text
YP_NE3A/
  AG1/good_1667
  AG1.5/good_1667
  AG2/good_1667
  AG2.5/good_1667
  AG3/good_1667
  AG4/good_1667
  AG5/good_1667
```

每个文本文件每行一个 RF，字段用空格或制表符分隔：

```text
NE3A_09276_072737.eqr  0.1  -100  401  amplitude_1 amplitude_2 ... amplitude_501
```

上面的振幅名称只是格式示意，真实输入必须全部是数值。

| 字段 | 含义 |
|---|---|
| 第 1 项 | `.eqr` 文件名，用于跨滤波视图对齐和输出名单 |
| 第 2 项 | 采样间隔 delta，单位秒 |
| 第 3 项 | 相对于直达 P 波零点的起始样点索引 |
| 第 4 项 | 结束样点索引，右端不包含 |
| 后续项 | 振幅，数量必须等于结束索引减起始索引 |

例如 `0.1 -100 401` 对应 501 点，时间为 −10.0 到 40.0 s。
索引不是秒；第 4 项也不是样点总数。支持普通小数、E/D 科学计数法和 UTF-8 BOM。
`good_1667`、`bad_1667` 的名称不作为预测标签或分数依据。

目录扫描识别 `good_N`、`bad_N`（可带 `.txt`）和 `.txt`、`.files`、`.dat` 文本。
也可直接传入任意命名的单个导出文件。台站内按 `.eqr` 文件名对齐，不要求各视图行顺序相同；
允许缺少部分视图。同一台站、同一事件、同一系数出现重复记录时，跳过整个事件并写明原因。
多台站输入应各自放在独立台站目录，避免把不同台站当成同一个 FCM 上下文。

## 单频、阈值、选择视图与重采样

```bash
# 多频，默认固定种子 20260928 / 阈值 0.90
rfqc-bench screen-eqr "inputs/YP_NE3A" --input-type files --output "results/record_multi"

# AG3 单频，默认固定种子 20260928 / 阈值 0.86
rfqc-bench screen-eqr "inputs/YP_NE3A" --input-type files --model reference_ag3 --output "results/record_ag3"

# 自定义阈值和参与判断的滤波视图；转换不同时间采样率
rfqc-bench screen-eqr "inputs/YP_NE3A" --input-type files --threshold 0.8 --filters 1 3 5 --resample --output "results/record_custom"

# 直接读取 AG3 下的单个导出文件
rfqc-bench screen-eqr "inputs/YP_NE3A/AG3/good_1667" --input-type files --model reference_ag3

# 没有 AG 目录时，明确实际高斯系数；一次只提供一个台站
rfqc-bench screen-eqr "/path/to/export.txt" --input-type files --model reference_ag3 --gaussian 3
```

`--filters` 指 Gaussian 系数，`--resample` 处理 delta 时间采样率，两者不同。
重采样使用与 SAC 输入相同的时间窗检查和转换方法，必须实际覆盖 −10 到 40 s，不能补造缺失时间。
阈值规则仍为 `p_good >= threshold`。其他纯波形方法可通过 `--model` 选择；
特征/combined 模型仍需提供六项描述量，不从文本中推断缺失特征。

## 输出及异常处理

默认在输入目录生成 `record`；输入单个文件时，在该文件父目录生成。
名单无表头、只写 `.eqr` 文件名，全局去重并去除所有目录前缀：

```text
NE3A_09276_072737.eqr
```

| 输出 | files 模式内容 |
|---|---|
| `record.predictions.csv` | 事件分数、判断、阈值、参与视图；`files` 是来源导出路径，`source_lines` 是对应行号 |
| `record.inputs.csv` | 每个有效波形视图的来源文件、行号、事件名、采样网格、重采样方式与原始行哈希 |
| `record.rejected.csv` | 无法解析、采样不合适、重复或未使用的输入及其文件/行号 |
| `record.json` | `input_type=files`、模型与阈值、数量、来源文件和输出哈希 |

文本数量统计：`files_found` 是导出文件数，`waveform_rows_found` 是候选非空行数，
`events_screened` 是对齐后实际判断的事件数，`record_entries` 是去重名单行数。
`retained_files`、`files_resampled` 在该模式统计波形视图行数，不是导出文件数。

原文本不改写。已有名单需换用 `--output`，或明确使用 `--overwrite`。
无效输入会写入异常清单；所有有效事件预测 bad 时名单正常为空，没有有效事件时返回退出码 2。
数值波形中必须保留文件名才能生成对应的 `.eqr` 名单；仅有振幅矩阵无法推断文件名。
文本没有 baz、射线参数等 SAC 头信息，筛选不会补造这些信息或执行 H–κ。

## Python 和 HTTP

```python
from rfqc_bench import RFQCPredictor, screen_eqr

report = screen_eqr("/path/to/YP_NE3A", input_type="files",
                    output="results/record", threshold=0.8, resample=True)
print(report["events_screened"], report["record_entries"])

model = RFQCPredictor.from_pretrained("reference_ag3")
model.screen_eqr("/path/to/YP_NE3A", input_type="files", output="results/record_ag3")
```

已有 `POST /screen-eqr` 接口增加 `input_type`，省略时仍为 `eqr`：

```bash
curl -X POST http://127.0.0.1:8000/screen-eqr -H "Content-Type: application/json" \
  -d '{"directory":"incoming/YP_NE3A","output":"results/record","input_type":"files","threshold":0.8,"resample":true}'
```

文件必须在服务器允许的 `--eqr-root` 内。`--max-eqr-files` 在 files 模式限制候选波形行数，
与 EQR 模式限制 `.eqr` 文件数对应。HTTP 的路径限制、输出冲突保护和模型锁仍适用。

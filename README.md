# 吟诗作赋任务测试框架

这是一个面向中文古诗词生成模型的模块化评测框架。当前支持四类任务：

- `theme_generation`：主题词生成，评价格式正确率。
- `keyword_generation`：关键词约束生成，评价关键词覆盖率，可选重复率。
- `prefix_continuation`：上句续写下句，评价 BLEU/ROUGE/字符重叠百分比。
- `style_control`：风格控制生成，并通过 LLM-as-a-Judge API 对测试模型生成的诗打分。

当前阶段程序内部已经内置测试样本构造逻辑，不需要手动输入单条样本，也不需要额外准备 jsonl 文件。

## 只改一个配置文件

模型种类、本地模型路径、生成参数、判分 API 都集中在：

```text
configs/model_config.yaml
```

本地待测模型配置：

```yaml
model:
  model_name: "Qwen2.5-0.5B-Instruct"
  model_type: "huggingface"
  model_path: "./models/Qwen2.5-0.5B-Instruct"
  tokenizer_path: "./models/Qwen2.5-0.5B-Instruct"
  device: "cuda"
```

如果只想快速跑通流程，可改为：

```yaml
model:
  model_name: "custom_poem_model"
  model_type: "mock"
  model_path: "./checkpoints/your_model_path"
  tokenizer_path: "./checkpoints/your_model_path"
  device: "cpu"
```

判分 API 配置也在同一个文件中：

```yaml
judge:
  api_url: "https://api.deepseek.com/chat/completions"
  api_key: "在这里填写你的真实 API Key"
  model_name: "deepseek-chat"
  require_api: true
```

把 `api_key` 改成你的真实 key 后，运行风格控制评测时就不需要再在命令行传 key。`api_url` 可以填写完整接口地址 `https://api.deepseek.com/chat/completions`；如果只填写 `https://api.deepseek.com`，脚本也会自动补齐 `/chat/completions`。`require_api: true` 表示必须调用真实判分 API；如果改为 `false` 且不填 `api_url`，脚本会使用 mock 判分，只适合检查流程。

注意：如果运行结果全是 `parse_error`，请先打开 `outputs/generations/style_control_results.csv` 查看 `error` 列。常见原因是：

- `judge.api_url` 写成了错误地址。
- `judge.api_key` 没有保存到 `configs/model_config.yaml`。
- API 余额、模型名或网络请求异常。

## 运行方式

在项目根目录 `poem_eval_framework` 下运行：

```bash
python evaluation/run_theme_eval.py
python evaluation/run_keyword_eval.py
python evaluation/run_prefix_eval.py
python evaluation/run_style_eval.py
```

启动 WebUI：

```bash
python webui.py
```

WebUI 默认不是示例数据：顶部“模型选择”模块会自动扫描 `models/` 文件夹中的子目录，例如 `models/Qwen2.5-0.5B-Instruct`。你后续把多个 Hugging Face 本地模型放进 `models/模型目录名/` 后，点击“刷新模型列表”即可在下拉框中选择。所选模型会覆盖 `configs/model_config.yaml` 里的 `model_path/tokenizer_path/model_name`，单条测试和批量本地评测都会使用当前选择的模型。界面中的评测结果会以 Markdown 表格展示关键指标，不再显示嵌套 JSON 调试结构。

WebUI 顶部提供“批量本地评测”折叠区，会直接调用以下本地评测入口对应的函数，输出仍保存到 `outputs/`：

```bash
python evaluation/run_theme_eval.py
python evaluation/run_keyword_eval.py
python evaluation/run_prefix_eval.py
python evaluation/run_style_eval.py
```

WebUI 的“上句续写”单条测试不再要求用户手动填写上句和参考答案，也不再要求选择五言绝句或七言绝句；点击开始测试时会从内置上句-答案样本池随机抽取一条，并在右侧指标表中显示抽中的上句和参考答案。

为控制 API 成本，WebUI 中的裁判大模型打分当前仅用于“风格控制”任务；主题词生成、关键词约束和上句续写只展示规则指标。

如果临时想覆盖配置文件中的判分 API，也可以传命令行参数：

```bash
python evaluation/run_style_eval.py ^
  --judge_url https://api.deepseek.com/chat/completions ^
  --judge_key YOUR_API_KEY ^
  --judge_model deepseek-chat ^
  --max_samples 100 ^
  --require_judge_api
```

风格控制任务默认生成并判分 100 首诗。可以用 `--max_samples` 调整数量，例如 `--max_samples 10` 只测试 10 首；`--max_samples 0` 只运行 20 条基础风格样本。

## 输出文件

结果默认保存到：

```text
outputs/generations/
outputs/metrics/
```

风格控制完整评测会生成：

```text
outputs/generations/style_control_results.csv
outputs/metrics/style_control_metrics.json
```

`style_control_results.csv` 包含测试样本、生成 prompt、测试模型生成诗、五个维度分数、重算后的 `total_score`、简短评价、失败类型和原始 judge 响应。

## 判分规则

LLM-as-a-Judge 会从五个维度给出 0 到 100 分：

- 格式正确性：是否符合指定诗体。
- 主题相关性：是否围绕主题或输入展开。
- 语言流畅性：是否自然、无乱码。
- 古诗风格：是否具有古诗词意象与表达。
- 意境创造性：是否有画面感和意境。

当前判分 prompt 采用偏严格的百分制标准：90 分以上只给明显优秀的输出，95 分以上应非常少见；如果句数/字数不符合诗体、偏题、风格不明显、出现现代解释性文字、乱码或明显重复，会被强制降档。百分制会比 1-5 分更容易拉开不同诗歌之间的差距。

代码会强制重算总分，覆盖判分模型返回的 `total_score`：

```text
total_score = 0.25 * format_score
            + 0.25 * theme_score
            + 0.20 * fluency_score
            + 0.20 * style_score
            + 0.10 * creativity_score
```

因此 `total_score` 也是 0 到 100 的百分制分数。

如果判分模型输出中夹杂分析文字、Markdown 代码块或多余内容，解析器会优先提取 ```json 代码块；没有代码块时，会尝试截取首个 `{` 到最后一个 `}`。解析失败时会返回 0 分兜底并标记 `failure_type=parse_error`。

## 生成清洗

主题词生成的 prompt 已明确要求模型只输出诗词正文，不输出标题、解释、赏析、序号或说明文字。对于 Qwen2.5-Instruct 这类聊天模型，主题词生成会优先使用 tokenizer 的 chat template 构造输入，减少“下面是……”等助手话术。

生成后还会做基础清洗：去除“好的”“下面是”“以下是”“标题”等常见前缀，并尽量只保留前四行诗句正文。格式是否真正合格仍由 `evaluation/format_check.py` 负责判定。

上句续写的 prompt 只要求模型根据给定上句续写下一句，不再提示“五言绝句/七言绝句”，避免模型把“续写下一句”误解为“补全整首绝句”。该任务还会使用专门的续写清洗逻辑，移除“下句：”“续写：”“答案：”等前缀、原上句、解释和多余行，只保留生成的下一句本身。核心指标是生成下句与参考答案之间的 BLEU/ROUGE 重叠度。



## 依赖安装

```bash
pip install -r requirements.txt
```

如果使用 Hugging Face 本地模型，需要 `transformers` 和 `torch`。如果使用 LLM-as-a-Judge，需要 `requests`；当前 `requirements.txt` 已包含常用依赖。

# LLM API Key 设置

`src/llm_client.py` 现在会**自动加载仓库根目录的 `.env`**（首次调用
`make_client()` 时）。不需要再手动 `load_dotenv()`，也不需要往
PowerShell 里 export。

## 方式 A：云端 DeepSeek（推荐评测用）

1. 到 https://platform.deepseek.com/api_keys 申请 key
2. 编辑仓库根目录的 `.env`，把占位符换成真 key：

```
DEEPSEEK_API_KEY=sk-你的真key
```

3. 验证：

```powershell
python -c "from src.llm_client import make_client, describe_llm_config; c=make_client(); print('enabled:', c.enabled); print(describe_llm_config())"
```

应输出 `enabled: True` 和 `hosted | base_url=... | api_key=yes`。

## 方式 B：本地自托管 vLLM（不需要真 key）

在 `.env` 里：

```
STENRAG_LLM_BASE_URL=http://127.0.0.1:8013/v1
STENRAG_LLM_MODEL=llama-3.1-8b
```

设了 `STENRAG_LLM_BASE_URL` 就会**优先**指向本地服务器，走占位 key
（vLLM 默认忽略 bearer token）。权重查找与自动启动见
`src/local_llm_server.py`。

## 单次运行临时覆盖

优先级是 **显式参数 > 真实环境变量 > `.env`**，所以环境变量不会被
`.env` 覆盖：

```powershell
$env:DEEPSEEK_API_KEY = "sk-临时key"; python eval/quick20_eval.py --use-llm
Remove-Item Env:\DEEPSEEK_API_KEY
```

## 安全注意

- `.env` 已在 `.gitignore` 里，**绝对不要 commit**。
- key 如果贴进了聊天记录，跑完建议去 DeepSeek 控制台 rotate 一次。
- 启动日志里会打印脱敏摘要（端点 / 模型 / 有无 key），**不会**打印 key
  本身。

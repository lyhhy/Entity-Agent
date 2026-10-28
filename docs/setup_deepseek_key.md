# DeepSeek API Key 设置（一次性，永久生效）

## 在 PowerShell 里执行（一次）

```powershell
[System.Environment]::SetEnvironmentVariable(
    'DEEPSEEK_API_KEY',
    'sk-你的key在这里',
    'User'
)
```

## 验证

新开一个 PowerShell 窗口：
```powershell
python -c "import os; print(bool(os.environ.get('DEEPSEEK_API_KEY')))"
```
应该输出 `True`。

## 安全注意

- 写到 User 级（不是 System 级）——只影响你的账户
- 这个 key **绝对不要 commit 到 git**。如果担心泄漏，我可以在跑完后帮你用 [System.Environment]::SetEnvironmentVariable 把 key 删掉
- key 已经被你贴在聊天记录里——建议你跑完后去 DeepSeek 控制台 rotate 一次（生成新 key，禁用旧 key）

## 跑 quick20_eval 启用 LLM

```powershell
cd C:\Users\zsf\Desktop\lyh\StenRAG
python eval\quick20_eval.py --seed 7 --tag tfidfA_llm_s7 --use_llm
```

## 跑完后清 key

```powershell
[System.Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', $null, 'User')
```

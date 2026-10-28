@echo off
rem API key is read from the environment (set DEEPSEEK_API_KEY before running).
if "%DEEPSEEK_API_KEY%"=="" (
  echo [FATAL] DEEPSEEK_API_KEY is not set.
  exit /b 1
)
set PYTHONUNBUFFERED=1
cd /d "C:\Users\zsf\Desktop\lyh\StenRAG"
python eval\_run_with_key.py eval\hyperedge_llm_eval.py --split val --max-examples 20 --seed 33 --use-llm 1>eval\results\_hyperedge_llm.log 2>eval\results\_hyperedge_llm.err
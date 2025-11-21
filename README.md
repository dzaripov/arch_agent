# Автоматическая оптимизация гиперпараметров под time-series forecasting с помощью LLM - написание агента.

LLM генерирует архитектуры, подбирая гиперпараметры, обучает модель Informer и выбирает лучший результат из записанных ею логов.

# Установка

```python
git clone github.com/dzaripov/arch_agent
cd arch_agent
uv sync
```
## Структура

- Informer2020 - модифицированный исходный код модели и датасет
- informer_tool.py - тул в формате OpenAI для вызова модели со всеми гиперпараметрами
- llm_optimization_informer.py - запуск ллм с тулом, идет оптимизация
- llm_optimization_informer_updated.py - запуск ллм с тулом с измененным промптом и логированием
- [logs_history.log] - логи подбора через ллм
- [optuna_llm_logs.db] - дашборд Optuna
- test_llm_with_tools.py - тестовый пример запуска легкой оптимизации функции `(x-3)**2 + (y-5)**2`
- optuna_module.py - подбор гиперпараметров через Optuna
- example.py - запуск подбора гиперпараметров через Optuna
- llm_serve.sh - запуск sglang для локального инференса моделей
- `pyproject.toml`, `uv.lock` - файлы зависимостей

## Пример запуска

uv sync
python llm_optimization_informer.py

Не забудьте добавить свой ключ `OPENROUTER_API_KEY` в файл `.env` корневой папки

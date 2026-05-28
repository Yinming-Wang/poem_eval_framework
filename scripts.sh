cd /home/ubuntu/Project/hw/poetry/final_presentation/poem_eval_framework
uv run python evaluation/run_theme_eval.py --max_samples 1000
uv run python evaluation/run_keyword_eval.py --max_samples 0
uv run python evaluation/run_prefix_eval.py --sample_source builtin --max_samples 0 --shuffle
uv run python evaluation/run_style_eval.py --max_samples 100
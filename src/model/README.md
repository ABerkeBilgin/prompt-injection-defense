# Model Module

Bu dizin artik first-party DefensiveToken akisina baglidir.

- `setup.py`: repo icindeki `src.official_stacks.defensivetoken` modulu ile savunmali modeli olusturur
- `run_inference.py`: baseline veya defended model ile yerel batch inference yapar
- `demo.py`: defended model icin kucuk ornek prompt akisini gosterir

Ana yol:
1. `python src/model/setup.py Qwen/Qwen2.5-7B-Instruct`
2. `python scripts/run_official_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite instruction --mode defense --judge official_api`

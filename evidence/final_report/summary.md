# NIAH results — metric `correct_word`

## Accuracy by context length

| arm | 4k | 8k | 16k | 32k | 64k | 128k | mean |
|---|---|---|---|---|---|---|---|
| base (native RoPE) | 100.0% | 100.0% | 100.0% | 100.0% |  66.7% |  33.3% |  91.2% |
| yarn f=4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Delta vs base (positive = YaRN helps)

| length | yarn f=4 |
|---|---|
| 4096 | +0.0 pp |
| 8192 | +0.0 pp |
| 16384 | +0.0 pp |
| 32768 | +0.0 pp |
| 65536 | +33.3 pp |
| 131072 | +66.7 pp |

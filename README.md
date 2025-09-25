# Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment

## Requirement & Installation

This repository is built upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Simply install all dependencies via:
```bash
conda create -n mlc python==3.10
conda activate mlc
pip install -r requirements.txt
```

For detailed environment setup (CUDA, deepspeed, vLLM, etc.), please also refer to the [LLaMA-Factory installation guide](https://github.com/hiyouga/LLaMA-Factory#installation).

## Dataset

The data has been placed in the `/data` directory and registered in `data_info.json`, specifically we use `multilin-pku-saferlhf-alpaca3-8b-train.json` for training.

## Training

Please run the following command to start the training process.

```sh
llamafactory-cli train examples/MLC/{model}.yaml
```

model = gemma2-9b / qwen2.5-7b

## Evaluation

We provide the safety evaluation data in the `/safe_eval`  directory. We adopt GPT-4o as the evaluation model and follow a deterministic decoding strategy with greedy sampling (temperature = 0, top-k = 1). The evaluation prompts are adapted from those used in the original papers corresponding to each dataset. For more details, please refer to Appendix.

We conduct general capability evaluation (MMLU, MMMLU-lite) on [Opencompass](https://github.com/open-compass/opencompass).

## Credits
The code of this repository relies on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and we would like to show the sincere gratitude to authors of it.

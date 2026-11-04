# RuMMSD: Russian Multimodal Sarcasm Detection

A comparative study of multimodal (image + text) sarcasm detection in **Russian** versus **English**.
Most multimodal sarcasm benchmarks are English-only, so this project builds a small, balanced Russian
image–text dataset and tests how well commercial LLMs and fine-tuned vision–language models transfer to it.

## Dataset

[`data/rummsd_ru.csv`](data/rummsd_ru.csv): **398 Russian social-media posts**, each paired with an image,
balanced at **199 sarcastic / 199 non-sarcastic**.

| column      | description                                   |
|-------------|-----------------------------------------------|
| `id`        | sample id                                     |
| `text`      | post caption (lower-cased, tokenized, ~15 words on average) |
| `label`     | `1` = sarcastic, `0` = not sarcastic          |
| `image_url` | direct download link to the paired image      |
| `image_id`  | image filename (`.jpg` / `.png`)              |

All experiments use a 70 / 30 train–test split with `random_state=42`.

## Approaches

| notebook | approach |
|----------|----------|
| [`01_gpt4o_mini_finetune_text.ipynb`](notebooks/01_gpt4o_mini_finetune_text.ipynb) | GPT-4o-mini fine-tuned through the OpenAI API on the text-only task, then evaluated on the held-out test set |
| [`02_gpt4_turbo_resnet_fusion.ipynb`](notebooks/02_gpt4_turbo_resnet_fusion.ipynb) | GPT-4 Turbo text representations fused with ResNet-18 image features in a trained PyTorch classifier |
| [`03_claude3_haiku_and_clip.ipynb`](notebooks/03_claude3_haiku_and_clip.ipynb) | Claude 3 Haiku (AWS Bedrock) zero- and few-shot prompting on image + text, including a two-step "describe the image, then judge" prompt; plus a CLIP ViT-B/32 fine-tuning baseline |

## Results

Fine-tuned GPT-4o-mini, text only, on 119 held-out samples:

| class         | precision | recall | F1   |
|---------------|-----------|--------|------|
| Not sarcastic | 0.65      | 0.95   | 0.77 |
| Sarcastic     | 0.92      | 0.54   | 0.68 |
| **Accuracy**  |           |        | **0.73** (macro F1 0.72) |

The model is precise when it predicts sarcasm (0.92) but misses almost half of the sarcastic posts.
This suggests that the text alone often does not carry the sarcasm: the cue is the contrast between the
caption and the image, which motivates the multimodal approaches in notebooks 02 and 03.

## Running

The notebooks were written for Google Colab.

```bash
pip install -r requirements.txt
```

- **OpenAI** (notebooks 01, 02): store your key as a Colab secret named `OPENAI_API_KEY`.
  Never hard-code it.
- **AWS Bedrock** (notebook 03): configure credentials with `aws configure` and enable
  Claude 3 Haiku access in `us-east-1`.
- **Images:** download them from the `image_url` column. Notebooks 02 and 03 contain helper functions for this.

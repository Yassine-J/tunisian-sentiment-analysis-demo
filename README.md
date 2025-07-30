# Tunisian Sentiment Analysis with Arabic-BERT 🇹🇳

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/repo/blob/main/ArabicBERT_Sentiment_Analysis.ipynb)

This project uses [ARABERT](https://huggingface.co/asafaya/bert-base-arabic), a BERT-based model pre-trained on Arabic dialects including Tunisian, to perform sentiment analysis on Tunisian Arabic text.

## Project Goals

- Demonstrate Tunisian dialect sentiment analysis using transformer models
- Provide an end-to-end pipeline from raw data to predictions
- Showcase transfer learning with Arabic-BERT for low-resource dialects

## What is Arabic-BERT?

Arabic-BERT is a family of BERT models specifically trained on Arabic text, including dialectal content. The base model has:

- 12 layers
- 768 hidden dimension
- 110M parameters
- Trained on 8.5B tokens of Arabic text

Available sizes:  
`mini (41M)` | `medium (110M)` | `base (110M)` | `large (335M)`

**Resources**:  
- [Model Card](https://huggingface.co/asafaya/bert-base-arabic)  
- [GitHub](https://github.com/alisafaya/Arabic-BERT)  

## Dataset

The dataset `sample.csv` contains Tunisian Arabic text with binary sentiment labels:

| Column | Description          |
|--------|----------------------|
| text   | Tunisian Arabic text |
| label  | 0=Negative, 1=Positive |

**Note**: if the csv file doesn't show the arabic words, do Data -> From Text/CSV -> UTF-8 -> Load

**Example Data**:
```csv
text,label
"ما عجبنيش الخدمة",0
"خدمتهم زينة والله",1
```
## Quick Start
```python
pip install transformers torch pandas
```
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model (replace with your fine-tuned path)
model_name = "asafaya/bert-base-arabic" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Predict example
inputs = tokenizer("ما نصحكش بهذا المطعم", return_tensors="pt")
outputs = model(**inputs)
```

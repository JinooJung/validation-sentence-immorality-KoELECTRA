# Introduction
This is 12th team's project page of 2023-1 "Introduction to Natural Language Processsing" Final term project.

Topic : Validation Immorality of Sentence using Ko-ELECTRA pretrained model.

Dataset : Korean Sentence Corpus "텍스트 윤리검증 데이터" from [AI Hub] 

Goal : To make Fine-tuned model with Ko-ELECTRA that validate senetences to Immoral or not, anc classify immorality type.

# Prerequisites
We run these codes on python 3.10.9

To install requirements, run this command on your environment.
```bash
pip install -r requirements.txt
```

# Preparing Dataset
Raw Dataset can be downloaded in [AI Hub] "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=558".

Set the path in rawdata_parser.py and run it.

And then, run format_converter.py that is fit to Dataset Class.

For training process, under code must run 2 times with Training data and Validation data. (Change file name in code)
```bash
python rawdata_parser.py
python format_converter.py
```

# Model
ELECTRA
- Paper : https://openreview.net/pdf?id=r1xMH1BtvB
- Github : https://github.com/google-research/electra

Ko-ELECTRA
- Github : https://github.com/monologg/KoELECTRA
- Huggingface : https://huggingface.co/monologg/koelectra-base-v3-discriminator

We used Ko-ELECTRA pretrained model and tokenizer from Huggingface.

```python
from transformers import ElectraTokenizer, ElectraForSequenceClassification
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=8)
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
```

# Training
Set configs in the main.py line 21-24, and run main.py.

Codes are only run on single GPU. If you want to run on multi GPU, you must change code.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

# Data Analysis
We tested our 2 model to sensitivity on intensity of immoral sentences.

test_ethic.py generates result_intensity_1.png and result_intensity_2.png.

You can check the image that is in presentation.

# Fine-tuned Model & Demo
You can try our model via demo.ipynb file.

First, download our fine-tuned model in this link : "https://drive.google.com/drive/u/2/folders/1lLZjN37ePJSktwUB_P4xq1vhalp65oAx"
We recommend to use mixup.ptl file. (It is trained with mixup augmentation)

Second, download demo.ipynb and set enviroment that is same with training process. (Using Google colab is also recommended.)

Third, set model path on the first cell, and run all cells sequentially. You can change "text" on the last cell and inference your own sentences.

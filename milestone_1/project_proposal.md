# Project proposal

### Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Engineering](#engineering)
- [Previous Works](#previous-works)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [References](#references)

---

### *Introduction*

The task we will work on is keyword spotting (KWS) in spoken language. The objective of this task is to detect utterances of specific keywords in audio recordings. KWS has many commercial applications, such as voice-enabled interfaces or customer service automation. 

KWS typically involves two parts: 1) embedding of the input audio sequence; and 2) a classification model with 3 categories—target (i.e. detected word is the desired keyword), unknown (i.e. detected word is not the desired keyword), and background noise (i.e. no words were detected). 

For our project, we will follow the works of Mazumder et al. (2021) to investigate how __multilingual embedding models__ can enable __few-shot keyword spotting__ in any language [1][2]. The (pre-)trained embedding model can be fine-tuned with just a small number of samples (e.g. 5) for each keyword, rather than a large dataset, for a relatively robust KWS system. The resulting model is especially useful for low-resource languages that may have limited amount of available data. 

---

### *Previous Works* 

#### [link to the project paper](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf)

Previous approaches to keyword spotting were time-consuming and required thousands of samples of the target keyword, as well as manual work. This project here used a simpler embedding scheme and considered few-shot performance across a larger number of languages and speakers, using only 5 training examples of a target keyword, which achieved high accuracy rates. 

#### [link to the project paper](https://openreview.net/pdf?id=c20jiJ5K2H)

Another project generated the Multilingual Spoken Words Corpus (which has over 340,000 keywords in 50 languages, while previous keyword spotting datasets tend to be small and limited to a single language) by using forced alignment techniques on crowdsourced audio data, and produced per-word timing estimates for extraction. In addition, the project provides a metric to evaluate extraction quality, and plans to regularly expand and update the dataset. 

---

### *Data* 

The dataset we will use is the __[Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/en/multilingual-spoken-words/)__. MSWC is a large and growing audio dataset of spoken words in 50 languages. The dataset contains more than 340,000 keywords for a total of 23.4 million 1-second spoken examples in `.opus` format, equivalent to more than 6,000 hours of audio recording. The size of the full dataset is 124 GB, and is freely available under the CC-BY 4.0 license. 

The dataset for each language can be downloaded separately. We will select a part of the languages for our project and store it in a shared Google Drive folder. Alternatively, we will explore the TensorFlow Datasets (TFDS) option. 

---

### *Engineering* 

We will follow 
#### Computing Infrastructure

We 
The computing infrastructure we will use is a combination of our personal laptops (CPU) and Google Colab. We will


#### Deep Learning of NLP 

- What ``deep learning of NLP (DL-NLP)`` methods will you employ? For example, will you do ``text classification with BERT?``, ``MT with T5``, ``language generation with transformers``, etc.? 

#### Framework 

- ``Framework`` you will use. Note: You *must* use PyTorch. Identify any ``existing codebase`` you can start off with. Provide links.

---

### *Evaluation* 

To evaluate the performance of models trained on the Multilingual Spoken Words Corpus extractions, we will use the `top-1 classification accuracy` as the primary metric, which will enable us to determine the percentage of correctly classified utterances and evaluate the model's overall classification performance.

In addition to the `top-1 classification accuracy`, we may also consider using the `False Reject Rate (FRR)` as a supplementary metric. FRR measures the proportion of true utterances that are incorrectly rejected by the model. By including this metric, we can further evaluate the model's ability to correctly classify all utterances.

As we construct a baseline model using CNN and potentially a Transformer-based model, we might utilize data visualization to compare and visualize the performances. The use of data visualization allows us to better understand the models' strengths and weaknesses.

--- 

### *Conclusion*

The purpose of the proposal is to establish a comprehensive framework for the project, including an introduction, relevant datasets, the specific models we will employ, and our chosen evaluation metric. This framework serves as a roadmap for our project, allowing us to learn from previous research and build upon existing knowledge to create model that incorporates our own unique features.
Our project proposal is subject to change... 

---

### *References*


[1] M. Mazumder, C. Banbury, J. Meyer, P. Warden, and V. J. Reddi. [Few-shot keyword spotting in
any language](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf). Proc. Interspeech 2021, 2021.

[2] M. Mazumder, S. Chitlangia, C. Banbury, Y. Kang, J. Ciro, K. Achorn, D. Galvez, M. Sabini, P. Mattson, D. Kanter, G. Diamos, P. Warden, J. Meyer, and V. J. Reddi. [Multilingual Spoken Words Corpus](https://openreview.net/pdf?id=c20jiJ5K2H). NeurIPS 2021, 2021.

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

#### [Mazumder et al. "Few-Shot Keyword Spotting in Any Language" (2021)](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf) [1] 

This paper proposes a method for automated extraction of large multilingual keyword bank from existing audio corpora. The multilingual keywordbank can then be used for training KWS classification models. The paper reports that multilingual embedding models outperform monolingual embedding models, regardless of the target language. The paper also demonstrates that the multilingual embedding model enables few-shot KWS models for unseen languages, performing relatively well with as few as 5-shots. 

#### [Mazumder et al. "Multilingual Spoken Words Corpus" (2021)](https://openreview.net/pdf?id=c20jiJ5K2H) [2]

Previously, most of the datasets in speech NLP were collected in controlled settings, in narrow domains, for one or two languages. This paper publishes a new dataset, the __[Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/en/multilingual-spoken-words/)__. The MSCW corpus provides a very large dataset for 50 languages, recorded in more natural settings. (Details of the dataset is discussed in the _**Data**_ section below.) In addition, the paper provides a metric for outlier detection, which can be used to filter out data samples with poor extraction quality. 

---

### *Data* 

The dataset we will use is the __[Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/en/multilingual-spoken-words/)__. MSWC is a large and growing audio dataset of spoken words in 50 languages. The dataset contains more than 340,000 keywords for a total of 23.4 million 1-second spoken examples in `.opus` format, equivalent to more than 6,000 hours of audio recording. The keywords were automatically extracted from the [Common Voice corpus](https://paperswithcode.com/dataset/common-voice) using TensorFlow Lite Micro's microfrontend spectrograms and Montreal Forced Aligners. The size of the full dataset is 124 GB, and is freely available under the CC-BY 4.0 license. 

The dataset for each language can be downloaded separately. We will select a part of the languages for our project and store it in a shared Google Drive folder. Alternatively, we will explore the TensorFlow Datasets (TFDS) option. 

---

### *Engineering* 

We will start by trying to reproduce the multinigual embedding + few-shot keyword spotting pipeline proposed by Mazumder et al. [1]. The authors released [code](https://github.com/harvard-edge/multilingual_kws) as well as a [Colab tutorial](https://colab.research.google.com/github/harvard-edge/multilingual_kws/blob/main/multilingual_kws_intro_tutorial.ipynb), but both of these are built with TensorFlow. Our project will instead try to build a Pytorch equivalent. 

The model will first embed audio inputs as 1024-dimensional vectors. A CNN layer with global average pooling will extract a 2048-dimensional vector from 49x40 spectograms of the audio input. The vectors will pass through two fully-connected ReLU with 2048 units and one fully-connected SELU activation layer with 1024 units. The strength of this model is that once the embedding model is ready, it can enable robust KWS through few-shot transfer learning. The fine-tuning process simply involves a Softmax layer of 3 categories, and we can expect to see results with as few as transfer learning with 5-shots.

The computing infrastructure we will use is a combination of our personal laptops (CPU) and Google Colab. After a more in-depth exploration, we will adjust the course of our experiment depending on how long it takes to train a model. It is unlikely that we will be able to exactly reproduce the authors' results, due to limitations in time and computational resources. However, we may be able to imitate a similar pipeline with smaller model size (i.e. less units), which can later be scaled up. We might also start with monolingual embedding, and examine how adding additional languages one by one affect the accuracy of downstream KWS tasks. 

<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/architecture.png" width=500/><br/>
  <b>Fig.1: KWS model architecture proposed by Mazumder et al. (2021) [1]</b>
</p>

---

### *Evaluation* 

To evaluate the performance of models trained on the Multilingual Spoken Words Corpus, we will use the `top-1 classification accuracy` as the primary metric, which will enable us to determine the percentage of correctly classified utterances and evaluate the model's overall classification performance [2].

In addition to the `top-1 classification accuracy`, we may also consider using the `False Reject Rate (FRR)` as a supplementary metric. FRR measures the proportion of true utterances that are incorrectly rejected by the model. By including this metric, we can further evaluate the model's ability to correctly classify all utterances [1][2].

As we construct a baseline model using CNN and potentially a Transformer-based model, we might utilize data visualization to compare and visualize the performances. For example, we might include ROC curves and FRR-FAR curves. The use of data visualization allows us to better understand the models' strengths and weaknesses.

--- 

### *Conclusion*

The purpose of the proposal was to map out our project framework, and to gain a deeper understanding about previous work and our dataset. We will use this proposal as the starting point of our project and experiment with our own alternative approaches. The specifics of our project is subject to change as we dive deeper into the next steps. 

---

### *References*

[1] M. Mazumder, C. Banbury, J. Meyer, P. Warden, and V. J. Reddi. [Few-shot keyword spotting in
any language](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf). Proc. Interspeech 2021, 2021.

[2] M. Mazumder, S. Chitlangia, C. Banbury, Y. Kang, J. Ciro, K. Achorn, D. Galvez, M. Sabini, P. Mattson, D. Kanter, G. Diamos, P. Warden, J. Meyer, and V. J. Reddi. [Multilingual Spoken Words Corpus](https://openreview.net/pdf?id=c20jiJ5K2H). NeurIPS 2021, 2021.

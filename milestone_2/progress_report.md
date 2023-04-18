# Project proposal

### Table of Contents

- [Introduction](#introduction)
- [Motivation, Contributions, Originality](#motivation-contributions-originality)
- [Previous Works](#previous-works)
- [Data](#data)
- [Engineering](#engineering)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [References](#references)

---

### *Introduction*

The task we will work on is keyword spotting (KWS) in spoken language. The objective of this task is to detect utterances of specific keywords in audio recordings. KWS has many commercial applications, such as voice-enabled interfaces or customer service automation. 

KWS typically involves two parts: 1) embedding of the input audio sequence; and 2) a classification model with 3 categories—target (i.e. detected word is the desired keyword), unknown (i.e. detected word is not the desired keyword), and background noise (i.e. no words were detected). 

For our project, we follow the works of Mazumder et al. (2021) to investigate how pre-trained __multilingual embedding models__ can enable __few-shot keyword spotting__ in any language [1][2]. The pre-trained embedding model can be fine-tuned with just a small number of samples (e.g. 5) for each keyword, rather than a large dataset, for a relatively robust KWS system. The resulting model is especially useful for low-resource languages that may have limited amount of available data. 

---

### *Motivation, Contributions, Originality*

Keyword spotting techniques have been widely utilized across a range of applications, providing benefits to individuals. Home devices like Google Home and Alexa have proven to be of great advantage, especially for those who are physically challenged, as speech-recognition technology enhances convenience in their daily lives. While English, as a high resource language, is not a challenging task, the application of keyword spotting techniques remains limited to other low resource languages. The applications are mature in academia and in the industry; however, we want to extend the techniques to other languages around the world using Pytorch. As a team comprising four multilingual speakers, we determined to leverage our strengths in expanding the use of keyword spotting techniques.

---

### *Previous Works* 

#### [Mazumder et al. "Few-Shot Keyword Spotting in Any Language" (2021)](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf) [1] 

This paper proposes a method for automated extraction of large multilingual keyword bank from existing audio corpora. The multilingual keywordbank can then be used for training KWS classification models. The paper reports that multilingual embedding models outperform monolingual embedding models, regardless of the target language. The paper also demonstrates that the multilingual embedding model enables few-shot KWS models for unseen languages, performing relatively well with as few as 5-shots. 

#### [Mazumder et al. "Multilingual Spoken Words Corpus" (2021)](https://openreview.net/pdf?id=c20jiJ5K2H) [2]

Previously, most of the datasets in speech NLP were collected in controlled settings, in narrow domains, for one or two languages. This paper publishes a new dataset, the __[Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/en/multilingual-spoken-words/)__. The MSCW corpus provides a very large dataset for 50 languages, recorded in more natural settings. (Details of the dataset is discussed in the _**Data**_ section below.) In addition, the paper provides a metric for outlier detection, which can be used to filter out data samples with poor extraction quality. 

#### [Zhu et al. “Locale Encoding for Scalable Multilingual Keyword Spotting” (2023)](https://arxiv.org/pdf/2302.12961.pdf) [3]

The paper mentions the limitations in previous works; the repeated monolingual approach does not exploit the common properties of acoustic data. To overcome this limitation, reduce the cost of training, and simplify the deployment process, the paper proposes a feature-wise linear modulation (FiLM) approach, which performed the best among baseline models such as locale-specific models where locale data was trained independently.

#### [Jung et al. “Metric Learning for User-defined Keyword Spotting” (2022)](https://arxiv.org/pdf/2211.00439.pdf) [4]

The paper highlights a limitation in earlier research that hinders its ability to apply to unseen terms. The paper proposes a metric learning-based training strategy to detect new spoken terms defined by users. To achieve this goal, the researchers collect large-scale out-of-domain keyword data, fine-tune the model, and perform ablations on the dataset.

#### [Deka & Das “Comparative Analysis of Multilingual Isolated Word Recognition Using Neural Network Models” (2022)](https://www.mililink.com/upload/article/806014499aams_vol_219_july_2022_a57_p5457-5467_brajen_kumar_deka_and_pranab_das.pdf) [5]

The present study performs a comparative analysis of the Scaled Conjugate Gradient algorithm for both Artificial Neural Network (ANN) and Recurrent Neural Network (RNN) models. The results indicate that RNN exhibits a superior performance in terms of the overall recognition rate when compared to ANN.

---

### *Data* 

The dataset we will use is the __[Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/en/multilingual-spoken-words/)__. MSWC is a large and growing audio dataset of spoken words in 50 languages. The dataset contains more than 340,000 keywords for a total of 23.4 million 1-second spoken examples in `.opus` format, equivalent to more than 6,000 hours of audio recording. The keywords were automatically extracted from the [Common Voice corpus](https://paperswithcode.com/dataset/common-voice) using TensorFlow Lite Micro's microfrontend spectrograms and Montreal Forced Aligners. The size of the full dataset is 124 GB, and is freely available under the CC-BY 4.0 license. 

The dataset for each language can be downloaded separately onto our local computer. We selected only a subset of the languages for our project, starting with English. 


<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/download-full.png" width=400/><br/>
  <b>Fig.1: Choosing between "Full dataset" and "Language" download options</b>
</p>

<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/download-language.png" width=400/><br/>
  <b>Fig.2: Downloading by language. a) `tar.gz` of `.opus` audio files organized by keyword;<br />b) `.csv` files containing the different splits of the monolingual dataset.</b>
</p>

If downloading by languages, the `.opus` audio files must be downloaded separately from the `.csv` split files (Fig.2). The audio files are organized by keywords. There are different number of samples for each keyword. Because the keywords are automatically extracted, the frequency of the samples roughly follow the word usage distribution of that language. 

```
. mscw/
├── en/                                      # English dataset   
│   ├── clips/                                   # Directory containing audio data
│   │   ├── aachen/                                  # Keyword 1 in English
│   │   │   ├── common_voice_en_18833718.opus            # Sample 1 of keyword 1 
│   │   │   └── ...                                      # More samples of keyword 1
│   │   ├── aalborg/                                 # Keyword 2 in English 
│   │   │   ├── common_voice_en_19552215__2.opus         # Sample 1 of keyword 2
│   │   │   └── ...                                      # More samples of keyword 2
│   │   └── ...                                      # More keywords in English 
│   ├── en_splits.csv                            # Joined .csv of all train/dev/test splits
│   ├── en_train.csv                                 # Table of files assigned to train split 
│   ├── en_dev.csv                                   # Table of files assigned to train split 
│   ├── en_test.csv                                  # Table of files assigned to train split 
│   └── version.txt                          # Version number of MSWC
└── ...                                      # More languages 
```

You can find an example of an `.opus` file [here](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/milestone_2/common_voice_en_10504.opus).

---

### *Engineering* 

We start by reproducing the multinigual embedding + few-shot keyword spotting pipeline proposed by Mazumder et al. [1]. The authors released examples of their [code](https://github.com/harvard-edge/multilingual_kws) as well as a [Colab tutorial](https://colab.research.google.com/github/harvard-edge/multilingual_kws/blob/main/multilingual_kws_intro_tutorial.ipynb), but both of these are built with TensorFlow. Our project instead builds a Pytorch equivalent. 

#### Data Preparation 

The full MSWC dataset is massive; consequently, we were only able to work on a subset of the data. To reduce the data size to a manageable scale, we chose smaller number of languages, smaller number of keywords, and limited the maximum number of samples for each keyword. These decisions resulted in three data preparation steps: 1) find the most frequent keywords within a single language dataset (i.e. keywords with most number of samples); 2) make a subset of the dataset to only include the desired keywords; and 3) prepare a Pytorch Dataset using the selected subset data.

[`keyword_ranker.py`](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/milestone_2/keyword_ranker.py)

`keyword_ranker.py` receives the `language code` and `number of keywords` as arguments. It counts and sorts the number of samples for each keyword in a single language dataset, then exports a list of as a `.txt` files [(see example here)](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/milestone_2/keywords_en_30.txt). 

[`keyword_selector.py`](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/milestone_2/keyword_selector.py)

The MSWC distributes several `.csv` files for each language dataset, which details which audio files are assigned to train, dev, or test splits. These splits are made so that samples of the same keyword is split in a ratio of 8:1:1, as well as with an even distribution of speaker's gender [2]. `keyword_selector.py.py` creates a slice of the `{LANG}_splits.csv` file to only include the top selected keywords pulled from `keyword_ranker.py`. Depending on our progress, we may extend this file so that it can merge keywords from several languages, for a multilingual embedding.

[`pytorch_prep.py`](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/milestone_2/pytorch_prep.py)

`pytorch_prep.py` imports the actual audio files, based on the sliced `.csv` file. If there are more than a certain number of samples for a keyword, we randomly select a subset of them (default of 1,000 samples). Then, we use [HuggingFace's WhisperFeatureExtractor](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperFeatureExtractor) and the [`librosa`](https://librosa.org/doc/latest/index.html) library to extract Pytorch tensors from the audio data. We decided on this feature extraction method after experimenting with several other methods, because it provided the most concise and lightweight features [(see `.ipynb` notebook here)](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/milestone_2/feature_extraction.ipynb). For the KWS classification task, we also converted the keywords corresponding to the audio files as indices. Finally, these Dataloaders are made from the Datasets, using a default batch size of 1024. These Dataloaders are exported for later use. 

<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/spectrogram.png" width=500/><br/>
  <b>Fig.3: Visualized spectrogram of our input tensor</b>
</p>

#### Next steps (subject to change)

The model will first embed audio inputs as 1024-dimensional vectors. A CNN layer with global average pooling will extract a 2048-dimensional vector from 49x40 spectograms of the audio input. The vectors will pass through two fully-connected ReLU with 2048 units and one fully-connected SELU activation layer with 1024 units. The strength of this model is that once the embedding model is ready, it can enable robust KWS through few-shot transfer learning. The fine-tuning process simply involves a Softmax layer of 3 categories, and we can expect to see results with as few as transfer learning with 5-shots.

The computing infrastructure we will use is a combination of our personal laptops (CPU) and Google Colab. After a more in-depth exploration, we will adjust the course of our experiment depending on how long it takes to train a model. It is unlikely that we will be able to exactly reproduce the authors' results, due to limitations in time and computational resources. However, we may be able to imitate a similar pipeline with smaller model size (i.e. less units), which can later be scaled up. We might also start with monolingual embedding, and examine how adding additional languages one by one affect the accuracy of downstream KWS tasks. 

<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/architecture.png" width=500/><br/>
  <b>Fig.4: KWS model architecture proposed by Mazumder et al. (2021) [1]</b>
</p>

---

### *Evaluation* 

To evaluate the performance of models trained on the Multilingual Spoken Words Corpus, we will use the `top-1 classification accuracy` as the primary metric, which will enable us to determine the percentage of correctly classified utterances and evaluate the model's overall classification performance [2].

In addition to the `top-1 classification accuracy`, we may also consider using the `False Reject Rate (FRR)` as a supplementary metric. FRR measures the proportion of true utterances that are incorrectly rejected by the model. By including this metric, we can further evaluate the model's ability to correctly classify all utterances [1][3].

As we construct a baseline model using CNN and potentially a Transformer-based model, we might utilize data visualization to compare and visualize the performances. For example, we might include ROC curves and FRR curves. The use of data visualization allows us to better understand the models' strengths and weaknesses.

--- 

### *Conclusion*

The purpose of the proposal was to map out our project framework, and to gain a deeper understanding about previous work and our dataset. We will use this proposal as the starting point of our project and experiment with our own alternative approaches. The specifics of our project is subject to change as we dive deeper into the next steps. 

---

### *References*

[1] M. Mazumder, C. Banbury, J. Meyer, P. Warden, and V. J. Reddi. [Few-shot keyword spotting in
any language](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf). Proc. Interspeech 2021, 2021.

[2] M. Mazumder, S. Chitlangia, C. Banbury, Y. Kang, J. Ciro, K. Achorn, D. Galvez, M. Sabini, P. Mattson, D. Kanter, G. Diamos, P. Warden, J. Meyer, and V. J. Reddi. [Multilingual Spoken Words Corpus](https://openreview.net/pdf?id=c20jiJ5K2H). NeurIPS 2021, 2021.

[3] P. Zhu, H. J. Park, A. Park, A. S. Scarpati, and I. L. Moreno. [Locale Encoding for Scalable Multilingual Keyword Spotting](https://arxiv.org/pdf/2302.12961.pdf). 2023.

[4] J. Jung, Y. Kim, J. Park, Y. Lim, B, Kim, Y, Jang, and J. S. Chung. [Metric Learning for User-defined Keyword Spotting](https://arxiv.org/pdf/2211.00439.pdf). 2022.
 
[5] B. K. Deka and P. Das. [Comparative Analysis of Multilingual Isolated Word Recognition Using Neural Network Models](https://www.mililink.com/upload/article/806014499aams_vol_219_july_2022_a57_p5457-5467_brajen_kumar_deka_and_pranab_das.pdf). Advances and Applications in Mathematical Sciences. 2022.

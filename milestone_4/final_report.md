
# Week 4: Final Report 

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#1-introduction)
- [Related Works](#2-related-works)
- [Datasets](#3-datasets)
- [Methods](#4-methods)
- [Results](#5-results)
- [Conclusion](#6-conclusion)
- [References](#references)

---

## *Abstract*

Inspired by the works of Mazumder et al. in multilingual embedding and keyword spotting, we compare the keyword spotting accuracy of monolingual and bilingual embedding models. Our final model used bilingual (English+Chinese) Whisper embedding and achieved embedding accuracy of 0.829. The KWS accuracy using this embedding model was 0.5767 for English words and 0.7055 for Chinese keywords.

---

## 1 *Introduction*

Keyword spotting (KWS) is an automatic speech recognition task to detect utterances of specific keywords in continuous audio recordings. KWS has many commercial applications, such as voice-enabled interfaces and customer service automation. Developing lightweight KWS models is important, especially for low-resource languages, where recorded voice data may not be abundant. 

KWS typically involves two parts: 1) embedding of the input audio sequence; and 2) a classification model with 3 categories—target (i.e. detected word is the desired keyword), unknown (i.e. detected word is not the desired keyword), and background noise (i.e. no words were detected). In this project, we were able to roughly recreate the works of Mazumder et al. (2021) in pre-trained __multilingual embedding models__ and __few-shot keyword spotting__ in any language [1][2], by using Whisper as our embedding model instead of EfficientNet, and implementing the model in Pytorch instead of TensorFlow. The pre-trained embedding model can be fine-tuned with just a small number of samples for each keyword, producing a relatively robust KWS system. Our final model used bilingual (English+Chinese) Whisper embedding and achieved embedding accuracy of 0.829. The KWS accuracy using this embedding model was 0.5767 for English words and 0.7055 for Chinese keywords.

---

## 2 *Related Works* 

The papers by Mazumder et al. [1][2] and Zhu et al. [3] focus on improving multilingual keyword spotting using various techniques. Mazumder et al. propose a method for extracting a large multilingual keyword bank from existing audio corpora, which outperforms monolingual models and enables few-shot KWS models for unseen languages [1]. They also publish a new dataset, the Multilingual Spoken Words Corpus (MSWC), extending an improved process of their previous work to 50 languages [2]. Zhu et al. propose a feature-wise linear modulation approach to overcome the limitations of repeated monolingual models and achieve better performance on multilingual KWS [3].

Some other previous works focus on various aspects of speech recognition and related tasks. Jung et al. propose a metric learning-based training strategy for detecting new spoken terms defined by users [4], while Deka & Das perform a comparative analysis of artificial neural network (ANN) and recurrent neural network (RNN) models for multilingual isolated word recognition [5]. Baevski et al. introduce the self-supervised learning framework, wav2vec2 [6], which is used in Berns et al. to fine-tune existing acoustic speech recognition networks for speaker and language change detection [7]. Mo et al. propose a neural architecture search approach to find convolutional neural network models for keyword spotting systems that achieve high accuracy while maintaining a small footprint[8]. Tan and Le introduce EfficientNet, a new CNN architecture that improves accuracy and efficiency through a compound scaling method [9].

---

## 3 *Datasets* 

In this section, we describe the two corpora that we used. 

### 3.1 Multilingual Spoken Words Corpus 

The main dataset we will use is the __[Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/en/multilingual-spoken-words/)__. MSWC is a large and growing audio dataset of spoken words in 50 languages. The dataset contains more than 340,000 keywords for a total of 23.4 million 1-second spoken examples in `.opus` format, equivalent to more than 6,000 hours of audio recording. The keywords were automatically extracted from the [Common Voice corpus](https://paperswithcode.com/dataset/common-voice) using TensorFlow Lite Micro's microfrontend spectrograms and Montreal Forced Aligners. The size of the full dataset is 124 GB, and the dataset is freely available under the CC-BY 4.0 license. 

The dataset for each language can be downloaded separately onto our local computer. We selected English and Chinese subsets for this subject.

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

<br/>

### 3.2 Google Speech Commands

Another dataset we will use is the __[Google Speech Commands (GSC)](https://arxiv.org/abs/1804.03209)__. GSC is an audio dataset consists of spoken words that are commonly used for enhancing the training and evaluation of KWS systems. It has 99.7k grouped keywords audio entries in `.wav` format. GSC also includes four background noise recordings. 

Mazumder et al. [1] proposed a technique to improve the performance of KWS systems by including 1-second samples of purely background noise during training. We follow their method of extracting random background noise samples from GSC, and incorporate them as 10% of each of the training, development, and test splits. To ensure consistency and simplicity, we used the pre-processed [Hugging Face](https://huggingface.co/datasets/speech_commands) version of the dataset.

```python
from datasets import load_dataset
dataset = load_dataset("speech_commands", "v0.02")
```
<p align="center">
  <b>Fig.3: How to import the GSC dataset from Hugging Face</b>
 </p>

---

## 4 *Methods* 

This section is divided into the embedding model and the keyword spotting task. 

### 4.1 Embedding Model

The embedding model serves as a feature extraction step that encodes the high-dimensional incoming audio features into a simpler set of features. This embedding layer will hopefully extract meaningful features from the audio that help distinguish between different keywords as well as the background noise. 

#### 4.1.1 Data Preparation 

##### 4.1.1.1 _Data Selection by Keyword_ `data_selection.py`

The full MSWC dataset is massive; consequently, we only work on a small subset of the data. To reduce the data size to a manageable scale, we chose only two languages, 30 keywords (based on frequency, i.e. the number of samples of keyword in the MSWC dataset), and limited the maximum number of samples for each keyword to 1,000 samples.

The script file `data_selection.py` receives the `language code` and `number of keywords` as arguments. It counts and sorts the number of samples for each keyword in a single language dataset, removes stopwords, then exports a list of keywords as a `.txt` files [(see example of .txt output here)](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/milestone_2/keywords_en_30.txt). 

The MSWC distributes several `.csv` files for each language dataset that details which audio files are assigned to train, dev, or test splits. These splits are made so that samples of the same keyword is split in a ratio of 8:1:1, as well as with an even distribution of speaker's gender [2]. `data_selection.py` creates a slice of the `{LANG}_splits.csv` file to only include the top selected keywords that was exported as a .txt file in the preivous step. 

In the case of bilingual (English+Chinese) embeddings, we manually combined the two `.csv` files generated by `data_selection.py` into one, and passed it onto `data_preparation.py` below. 

##### 4.1.1.2 _Pytorch Preparation_ `data_preparation.py`

The data preparation script `data_preparation.py`(https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/main/milestone_4/data_preparation.py) imports the audio files based on the sliced `.csv` file from `data_selection.py`. If there are more than a certain number of samples for a keyword, we randomly select a subset of them (up to 1,000 samples). Then, we use [HuggingFace's WhisperFeatureExtractor](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperFeatureExtractor) and the [`librosa`](https://librosa.org/doc/latest/index.html) library to extract Pytorch tensors from the audio data. We cropped the returned features to only include the 1-second interval. We experimented with several other feature extractors as well--you can check out the results in the [.ipynb notebook here](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/milestone_2/feature_extraction.ipynb). 

<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/spectrogram.png" width=500/><br/>
  <b>Fig.4: Visualized spectrogram of our input tensor</b>
</p>

The final version of the data preparation used Whisper with 80 channels as the embedding model. This was chosen after trying different embedding models, as described in the next section. After creating the features for each keyword audio file, the script also access the GSC via HuggingFace to extract random noise samples and include into our datasets. 

We also converted the keywords corresponding to the audio files as indices. Finally, these Dataloaders are made from the Datasets, and exported for later use. All of the Dataloaders can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/19un1Briw_Hobvu8HMnPuYCJvhEbzyX5H?usp=sharing).

<br/>

### 4.1.2 Building the Embedding Model 

We compared three different approaches for the embedding model: 1) EfficientNet; 2) Wav2Vec2; and 3) Whisper. 

<p align="center"><b>Table 1: Sitemap for Preparation Code</b></p>
<table align="center">
    <tr>
        <th>Model</th>
        <th>Code</th>
    </tr>
    <tr align="center">
        <td>EfficientNet</td>
      <td><a href="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/main/milestone_3/cnn_model.ipynb">cnn_model.ipynb</a></td>
    </tr>
    <tr align="center">
        <td >Wav2Vec2</td>
      <td><a href="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/main/milestone_3/wav2vec.ipynb">wav2vec.ipynb</a></td>
    </tr>
    <tr align="center">
        <td>Whisper</td>
      <td><a href="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/main/milestone_4/embedding.ipynb">whisper.ipynb</a></td>
    </tr>
</table>

__4.1.2.1 _EfficientNet___

The purpose of this approach was to understand the model made by Mazumder et al. [1]. We replicated the overall structure, but refactored the model into smaller dimensions to be slightly smaller than the Whisper approach. 

<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/architecture.png" width=500/><br/>
  <b>Fig.5: KWS model architecture proposed by Mazumder et al. (2021) [1]</b>
</p>

```python
class MultilingualEmbeddingModel(nn.Module):
    def __init__(self, num_classes):
        super(MultilingualEmbeddingModel, self).__init__()
        # Load EfficientNet-B0 as the base model
        self.efficient_b0_model = torchvision.models.efficientnet_b0(pretrained=True)

        # Add a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((None, 512))

        # Add two dense layers of 2048 units with ReLU activations
        self.linear1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        # Add a penultimate 1024-unit SELU activation layer
        self.linear3 = nn.Linear(512, 256)
        self.selu = nn.SELU()
        # add a softmax layer
        self.linear4 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
```

<p align="center">Fig.6: Our imitation of the architecture in Fig.5, with smalle dimensions</p>

__4.1.2.2 _[Wav2Vec](https://huggingface.co/docs/transformers/model_doc/wav2vec2)___

Wav2Vec2 is a speech model designed for self-supervised learning of audio dataset representations. The model comprises four essential elements: the feature encoder, context network, quantization module, and contrastive loss. Wav2Vec2 can achieve a satisfying WER score and perform speech recognition tasks with small amounts of labeled data, as little as 10 minutes. Although Wav2Vec2 is known to perform well on the Libri Speech dataset, its reputation is now replaced by Whisper. 

Our implementation of the Wav2Vec2 model can be found in [`wav2vec.ipynb`](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/main/milestone_3/wav2vec.ipynb).

__4.1.2.3 _[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)___

The Whisper is an encoder-decoder Transformer with an end-to-end approach. The pre-trained model generalizes well to standard benchmarks and achieves satisfying results without requiring fine-tuning. Whisper is the current state-of-the-art standard in automatic speech recogintion tasks. 

Our implementation of the Whisper model can be found in [`whisper.ipynb`](https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/main/milestone_3/whisper.ipynb).

<br/>

### 4.2 Keyword Spotting 

Mazumder et al. claimed that multilingual embeddings can be fine-tuned for keyword spotting with as little as 5 new samples [1]. The authors also stated that multilingual embeddings can help improve KWS performance in both high and low-resource languages, as well as language known and unknown to the embedding model [1]. 

We built a simple KWS classifier that received outputs from the embedding model as input. The classifier output had 3 classes--target, non-target, and background noise.

```python
class KWS_classifier(nn.Module):
    def __init__(self, input_size=31, output_size=3):
        super(KWS_classifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x
```
<p align="center">Fig. 7: Simple KWS classifier</p>

We trained three different embedding models: monolingual English, monolingula Chinese, and bilingual (English+Chinese). The pre-trained embedding models can be loaded from HuggingFace ([Monolingual English](https://huggingface.co/jaeihn/kws_embedding); [Monolingual Chinese](https://huggingface.co/jaeihn/kws_embedding_cn30); [Bilingual (English+Chinese)](https://huggingface.co/jaeihn/kws_embedding_bilingual)). We then trained/tested KWS on 10 English words and 10 Chinese words, to see how each embedding model affected the accuracy of the downstream KWS task. The results are summarized in the next section.

---

## 5 *Results*

The quality of the embedding model can be measured in terms of accuracy, since the embedding model is obtained from training a classification task during which audio features of different words yield corresponding words (or background noise). 

The performance of the KWS task can also be measure in terms of accuracy--whether the embedded input is correctly classified as target, non-target, or background noise. We compare the performance of KWS using monolingual and bilingual embeddings. 

### 5.1 Embedding Model

<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/wav2vec_experiments.png" /><br/>
  <b>Fig.8: Experiments with Wav2Vec2 as the embedding model</b>
</p>

<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/whisper_experiments.png" /><br/>
  <b>Fig.9: Experiments with Whisper as the embedding model</b>
</p>

<br/>

<p align="center">Table 2: Accuracy of Embedding Models</p>
<table align="center">
  <tr align="center">
    <th>Model</th>
    <th>Monolingual English Embedding</th>
    <th>Monolingual Chinese Embedding</th>
    <th>Bilingual (English+Chinese) Embedding</th>    
  </tr>
  <tr align="center">
    <td>EfficientNet</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td>Wav2Vec2</td>
    <td>0.313</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr align="center">
    <td>Whisper</td>
    <td>0.664</td>
    <td>0.753</td>
    <td>0.829</td>
  </tr>
</table>

<p align="center">
  <img src="https://github.ubc.ca/jaeihn/COLX_585_The-Wild-Bunch/blob/jae/screenshots/embedding_acc.png" width=500 /><br/>
  <b>Fig.10: Comparison of monolingual and bilingual embedding models</b>
</p>

<br/>

### 5.2 Keyword Spotting

<p align="center">Table 3: Accuracy of Keyword Spotting using Different Embedding Models</p>
<table>
  <tr align="center">
    <td></td>
    <th>Monolingual English Embedding</th>
    <th>Monolingual Chinese Embedding</th>
    <th>Bilingual (English+Chinese) Embedding</th>
  </tr>
  <tr align="center">
    <th>English KWS Accuracy</th>
    <td>0.6264</td>
    <td>0.5192</td>
    <td>0.5767</td>
  </tr>
  <tr align="center">
    <th>Chinese KWS Accuracy</th>
    <td>0.6897</td>
    <td>0.6997</td>
    <td>0.7055</td>
  </tr>
</table>

--- 

## 6 *Conclusion*

### 6.1 Discussion 

In conclusion, we found that Whisper outperformed Wav2Vec2 for building embedding models (Table 2). For monolingual embedding models, we found that the performance of the downstream KWS was better when the embedding model language matched the keyword language. For bilingual embedding models, the accuracy was higher than the monolingual Chinese embedding model for English keywords, and higher than both the monolingual embedding models for Chinese keywords. 

### 6.2 Challenges 

The biggest challenge in our project was the magnitude of speech datasets. It was time-consuming to download, convert, process, and model these dataset. We had to greatly downsize both the data subset and model parameters for it to be feasible. Experimenting with different parameters for the models quickly became difficult to manage, but we remedied this by using Weights and Biases (wandb) for automatic logging. 

### 6.3 Future work

First, we would like to complete the training for the embedding model based on EfficientNet, and compare the downstream KWS accuracy against that from the Whisper embedding model. For this project, the training process took too long for us to complete. This will allow us to roughly compare the performance of the original model by Mazumder et al., when the model is restricted to the smaller data subset and model parameters [1]. 

We could also add additional languages, one by one, and compare the downstream KWS accuracy between multilingual embedding models that uses combination of different languages. Mainly, we would like to attest original authors' claim that multilingual embedding model can help improve KWS accuracy of all languages. 

Finally, we would like to experiment with more languages for KWS. We would like to see how the performance varies with seen and unseen languages (i.e. languages included in the embedding model or otherwise), and if there are patterns in accuracies based on how "close" different languages are phonologically. 

---

## *References*

[1] M. Mazumder, C. Banbury, J. Meyer, P. Warden, and V. J. Reddi. [Few-shot keyword spotting in
any language](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf). Proc. Interspeech 2021, 2021.

[2] M. Mazumder, S. Chitlangia, C. Banbury, Y. Kang, J. Ciro, K. Achorn, D. Galvez, M. Sabini, P. Mattson, D. Kanter, G. Diamos, P. Warden, J. Meyer, and V. J. Reddi. [Multilingual Spoken Words Corpus](https://openreview.net/pdf?id=c20jiJ5K2H). NeurIPS 2021, 2021.

[3] P. Zhu, H. J. Park, A. Park, A. S. Scarpati, and I. L. Moreno. [Locale Encoding for Scalable Multilingual Keyword Spotting](https://arxiv.org/pdf/2302.12961.pdf). 2023.

[4] J. Jung, Y. Kim, J. Park, Y. Lim, B, Kim, Y, Jang, and J. S. Chung. [Metric Learning for User-defined Keyword Spotting](https://arxiv.org/pdf/2211.00439.pdf). 2022.
 
[5] B. K. Deka and P. Das. [Comparative Analysis of Multilingual Isolated Word Recognition Using Neural Network Models](https://www.mililink.com/upload/article/806014499aams_vol_219_july_2022_a57_p5457-5467_brajen_kumar_deka_and_pranab_das.pdf). Advances and Applications in Mathematical Sciences. 2022.

[6] A. Baevski, H. Zhou, A. Mohamed, and M. Auli. [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477.pdf). 2020.

[7] T. Berns, N. Vaessen, and D. A. Leeuwen. [Speaker and Language Change Detection using Wav2vec2 and Whisper](https://arxiv.org/pdf/2302.09381.pdf). 2023

[8] T. Mo, Y. Yu, M. Salameh, D. Niu, and S. Jui. [Neural Architecture Search for Keyword Spotting](https://arxiv.org/pdf/2009.00165.pdf). 2021.

[9] M. Tan, and Q. V. Le. [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). 2019.

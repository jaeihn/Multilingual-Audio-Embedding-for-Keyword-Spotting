# Project proposal

### Table of Contents

- [Previous Works](#previous-works)

### *Previous Works* 

#### [Mazumder et al. "Few-Shot Keyword Spotting in Any Language" (2021)](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf) [1] 

This paper proposes a method for automated extraction of large multilingual keyword bank from existing audio corpora. The multilingual keywordbank can then be used for training KWS classification models. The paper reports that multilingual embedding models outperform monolingual embedding models, regardless of the target language. The paper also demonstrates that the multilingual embedding model enables few-shot KWS models for unseen languages, performing relatively well with as few as 5-shots.

#### [Mazumder et al. "Multilingual Spoken Words Corpus" (2021)](https://openreview.net/pdf?id=c20jiJ5K2H) [2]

Previously, most of the datasets in speech NLP were collected in controlled settings, in narrow domains, for one or two languages. This paper publishes a new dataset, the __[Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/en/multilingual-spoken-words/)__. The MSCW corpus provides a very large dataset for 50 languages, recorded in more natural settings. (Details of the dataset is discussed in the _**Data**_ section below.) In addition, the paper provides a metric for outlier detection, which can be used to filter out data samples with poor extraction quality.

#### [Zhu et al. “Locale Encoding for Scalable Multilingual Keyword Spotting” (2023)](https://arxiv.org/pdf/2302.12961.pdf) [3]

The paper mentions limitations in previous works: the repeated monolingual approach does not exploit the common properties of acoustic data. To overcome this limitation, reduce the cost of training, and simplify the deployment process, the paper proposes the feature-wise linear modulation (FiLM) approach, which performed the best among baseline models such as locale-specific models where locale data was trained independently.

#### [Jung et al. “Metric Learning for User-defined Keyword Spotting” (2022)](https://arxiv.org/pdf/2211.00439.pdf) [4]

The paper highlights a limitation in earlier research that hinders its ability to apply to unseen terms. The paper proposes a metric learning-based training strategy to detect new spoken terms defined by users. To achieve this goal, the researchers collect large-scale out-of-domain keyword data, fine-tune the model, and perform ablations on the dataset.

#### [Deka & Das “Comparative Analysis of Multilingual Isolated Word Recognition Using Neural Network Models” (2022)](https://www.mililink.com/upload/article/806014499aams_vol_219_july_2022_a57_p5457-5467_brajen_kumar_deka_and_pranab_das.pdf) [5]

The present study performs a comparative analysis of the Scaled Conjugate Gradient algorithm for both Artificial Neural Network (ANN) and Recurrent Neural Network (RNN) models. The results indicate that RNN exhibits a superior performance in terms of the overall recognition rate when compared to ANN.

---

### *References*

[1] M. Mazumder, C. Banbury, J. Meyer, P. Warden, and V. J. Reddi. [Few-shot keyword spotting in
any language](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf). Proc. Interspeech 2021, 2021.

[2] M. Mazumder, S. Chitlangia, C. Banbury, Y. Kang, J. Ciro, K. Achorn, D. Galvez, M. Sabini, P. Mattson, D. Kanter, G. Diamos, P. Warden, J. Meyer, and V. J. Reddi. [Multilingual Spoken Words Corpus](https://openreview.net/pdf?id=c20jiJ5K2H). NeurIPS 2021, 2021.

[3] P. Zhu, H. J. Park, A. Park, A. S. Scarpati, and I. L. Moreno. [Locale Encoding for Scalable Multilingual Keyword Spotting](https://arxiv.org/pdf/2302.12961.pdf). 2023.

[4] J. Jung, Y. Kim, J. Park, Y. Lim, B, Kim, Y, Jang, and J. S. Chung. [Metric Learning for User-defined Keyword Spotting](https://arxiv.org/pdf/2211.00439.pdf). 2022.

[5] B. K. Deka and P. Das. [Comparative Analysis of Multilingual Isolated Word Recognition Using Neural Network Models](https://www.mililink.com/upload/article/806014499aams_vol_219_july_2022_a57_p5457-5467_brajen_kumar_deka_and_pranab_das.pdf). Advances and Applications in Mathematical Sciences. 2022.


# ScienceQA Dataset – Classification and Retrieval Project
This project aims to apply various Natural Language and Processing techniques on a Dataset.
We worked with the ScienceQA dataset, which contains over 20,000 science-related questions from elementary and high school curricula. Each record includes fields such as:
- image : contextual image
- question : the main prompt
- choices : multiple-choice answers
- answer : index of the correct choice
- hint : clue related to the question
- task : description of the task
- grade : grade level (K–12)
- subject : main subject area
- topic : broad category (natural sciences, social sciences, language sciences)
- category : more specific topic
- skill : skill being assessed
- lecture : passage related to the question
- solution : explanation of the correct answer

The dataset is already divided into training, validation and test sets. Our focus in this project was **text classification**, in particular predicting the topic of each question.

## Dataset Analysis

The dataset covers a wide variety of subjects, but the distribution of topics is very unbalanced. Some subjects such as biology and physics appear very frequently, while others are rare. This imbalance turned out to be one of the main challenges for our models. 

File : preliminary_analysis.ipynb

## Clustering and Topic Modeling 
### K-means
We first applied k-means clustering to explore whether questions naturally grouped together. While there were some clusters, the 3D visualization of high-dimensional data was not very clear.
### Topic modeling
We then tried topic modeling, which assigns each question a distribution over topics. This gave a more meaningful view of thematic groupings, with each document colored by its dominant topic.

File : clustering.ipynb

## Retrieval with TF-IDF and BM25

We built a simple search index over the questions and tested two classical retrieval methods (TF-IDF and BM25).
Both methods performed poorly (MAP ≈ 0.022, NDCG ≈ 0.048). This result is explained by the fact that questions in the same topic often use very different wording, so purely lexical similarity was not enough.

File : NLP_ScienceQA_David.ipynb

## Embeddings and Visualization

We then embedded the questions and visualized them with t-SNE. The embeddings captured semantic similarity much better than word-based indexes. Using these embeddings, a logistic regression classifier achieved strong performance, showing that vector representations were more suitable for this dataset.

File : section1_NLP_with_WE_classif.ipynb

## Classification Models
### LSTM

We trained an LSTM model on the question text. On the validation set it reached about 93.5% accuracy. The model worked well for frequent topics such as biology and physics, but almost never predicted rare topics correctly. This highlighted the effect of class imbalance.

### Transformer : BERT

To compare fairly, we always used the same backbone and only varied the input fields:

- Three-topic model, using question + answer + hint + solution + lecture
  - Accuracy: 92%
  - Main issue: biology and physics were often predicted as geography.

- Seven-topic model with the same rich input
  - Accuracy dropped to 24% (random baseline ≈ 14%).
  - The model tended to predict nearly everything as geography.

- Seven-topic model using only the question text
  - Accuracy: about 90%
  - Showed that using only the most informative field worked better than including too much context.

## Few Short Learning for Large Language Models (Flan-T5) 

We tested Flan-T5 Large for zero-, one- and few-shot classification:

**Zero-shot** on 50 random samples: ~90% accuracy.

**One- and few-shot**: performance dropped when we included long full-text examples, since the prompt became too long.

After limiting each example to 250 characters, accuracy improved significantly. With 1–2 shots performance was good, but it decreased again when the prompt grew longer.

File : transformer.ipynb

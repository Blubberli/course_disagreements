#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.special import rel_entr
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import f1_score
from tabulate import tabulate


# ## Assignment 2: Evaluation
# 
# This notebook implements the evaluation metrics discussed in the second part of the tutorial and looks at some example data. First we need a function that calculates the Jensen-Shannon divergence and cross-entropy between two probability distributions.

# In[ ]:


def jenson_shannon_divergence(human_dist, predicted_dist):
    """
    Calculate the Jensen-Shannon divergence between two probability distributions.
    :param human_dist: a probability distribution
    :param predicted_dist: a probability distribution
    :return: the Jensen-Shannon divergence between the two distributions
    """
    # Calculate the mid distribution
    average_distribution = 0.5 * (human_dist + predicted_dist)
    # kl-div between human and average
    kldiv_a = sum(rel_entr(human_dist, average_distribution))
    # kl-div between predicted and average
    kldiv_b = sum(rel_entr(predicted_dist, average_distribution))
    # Calculate the Jensen-Shannon divergence
    divergence = 0.5 * (kldiv_a + kldiv_b)

    return divergence


# In[ ]:


def cross_entropy(human_dist, predicted_dist):
    """
    Calculate the cross-entropy between two probability distributions.
    :param human_dist: a probability distribution
    :param predicted_dist: a probability distribution
    :return: the cross-entropy between the two distributions
    """
    # Calculate the cross-entropy
    ce = -sum(rel_entr(human_dist, predicted_dist))

    return ce


# Next we can calculate the Jensen-Shannon divergence and cross-entropy between the human and predicted distributions for differenet scenarios, as defined below.

# In[ ]:


# some example distributions for human data
human_distribution = {
"humans low uncertainty" : [0.9, 0.05, 0.05],
"humans moderate uncertainty" : [0.5, 0.3, 0.2],
"humans high uncertainty": [0.33, 0.33, 0.33],
"humans medium uncertainty - one dominant":  [0.7, 0.2, 0.1],
"humans low uncertainty - one dominant": [0.8, 0.1, 0.1],
}

predicted_distributions = {
"prediction high certainty wrong class": [0.05, 0.9, 0.05],
"prediction high certainty correct class": [0.9, 0.05, 0.05],
"prediction moderate certainty wrong class": [0.4, 0.5, 0.1],
"prediction max uncertainty": [0.33, 0.33, 0.33],
"prediction moderate certainty correct class": [0.5, 0.3, 0.2],
    
}


# 

# A1: Calculate the Jensen-Shannon divergence and cross-entropy between the human and predicted distributions for all possible cases. Which cases lead to a very low JS-divergence and cross-entropy? What does this mean? Which cases lead to a very high JS-divergence and cross-entropy? What does this mean? In which cases do JS-divergence and cross-entropy differ significantly? 

# Next we look at example data and example model predictions. We want to use different metrics to evaluate to what extent the model predictions agree with the human annotations.

# In[ ]:


def calculate_f1_per_annotator(annotations, model_predictions):
    """
    Calculate the F1 score for each annotator.
    :param annotations: a dataframe with the annotations for each annotator.
    :param model_predictions: a vector with the model predictions.
    :return: a list with the F1 score for each annotator. The list has length equal to the number of annotators.
    """
    f1_scores = []
    for annotator in annotations.columns:
        f1_scores.append(f1_score(annotations[annotator], model_predictions > 0.5))
    return f1_scores


# In[ ]:


def normalized_entropy(prob_dist):
    """
    Calculate the normalized entropy of a probability distribution.
    :param prob_dist: a probability distribution, e.g. a human distribution over 2 or more classes.
    :return: normalized entropy of the probability distribution, value between 0 and 1.
    """
    # Calculate entropy with base 2 (in bits)
    ent = entropy(prob_dist, base=2)
    # Max entropy occurs when all categories are equally likely
    max_ent = np.log2(len(prob_dist))
    # Normalize the entropy (divide by max possible entropy) take into account that entropy is 0 when all are the same
    if max_ent == 0:
        norm_entropy = 0
    else:
        norm_entropy = ent / max_ent
    return norm_entropy


# In[ ]:


def entropy_similarity(human_entropy_scores, model_entropy_scores):
    """
    Calculate the entropy similarity between two probability distributions.
    :param human_entropy_scores: a probability distribution
    :param model_entropy_scores: a probability distribution
    :return: the entropy similarity between the two distributions
    """
    # Calculate the cosine similarity between these two vectors
    ent_sim = np.dot(human_entropy_scores, model_entropy_scores) / (
            np.linalg.norm(human_entropy_scores) * np.linalg.norm(model_entropy_scores)
    )

    return ent_sim


# In[ ]:


def get_normalized_human_distribution(annotated_labels, num_classes):
    """
    Calculate the normalized human distribution.
    :param annotated_labels: the annotated labels
    :param num_classes: the number of classes
    :return: the normalized human distribution
    """
    # Calculate the histogram of the annotated labels
    hist, _ = np.histogram(annotated_labels, bins=num_classes, range=(0, num_classes))
    # Normalize the histogram
    normalized_human_distribution = hist / len(annotated_labels)

    return normalized_human_distribution


# In[ ]:


def entropy_correlation(human_entropy_scores, model_entropy_scores):
    """
    Calculate the entropy correlation between two probability distributions.
    :param human_entropy_scores: a probability distribution
    :param model_entropy_scores: a probability distribution
    :return: the entropy correlation between the two distributions
    """
    # Calculate the pearson correlation between these two vectors
    corr = np.corrcoef(human_entropy_scores, model_entropy_scores)[0, 1]

    return corr


# Now we can simulate some example data and model predictions and calculate different metrics for both datasets, both models (Model A and Model B).

# In[ ]:


np.random.seed(42)

# 100 samples with 3 annotators providing binary labels (0 or 1)
annotations_3_annotators = pd.DataFrame({
    'Annotator_1': np.random.randint(0, 2, 100),
    'Annotator_2': np.random.randint(0, 2, 100),
    'Annotator_3': np.random.randint(0, 2, 100)
})
annotations_3_annotators['human_distribution'] = [get_normalized_human_distribution(annotations_3_annotators.iloc[i], 2) for i in range(annotations_3_annotators.shape[0])]
# Simulate predictions by two models (Model A and Model B) for 3 annotators
model_A_predictions_3 = np.random.uniform(0.3, 0.7, 100)  
model_B_predictions_3 = np.random.uniform(0.2, 0.8, 100)  
annotations_3_annotators["MODEL_A"] = model_A_predictions_3
annotations_3_annotators["MODEL_B"] = model_B_predictions_3
# convert them into a probability distribution by putting the probability into index 1, if it is > 0.5 and into index 0 if it is <= 0.5, and add 1- probability to the other index
annotations_3_annotators["MODEL_A"] = annotations_3_annotators["MODEL_A"].apply(lambda x: [1-x, x] if x <= 0.5 else [x, 1-x])
annotations_3_annotators["MODEL_B"] = annotations_3_annotators["MODEL_B"].apply(lambda x: [1-x, x] if x <= 0.5 else [x, 1-x])

# Simulate the dataset for 5 annotators
annotations_5_annotators = pd.DataFrame({
    'Annotator_1': np.random.randint(0, 2, 100),
    'Annotator_2': np.random.randint(0, 2, 100),
    'Annotator_3': np.random.randint(0, 2, 100),
    'Annotator_4': np.random.randint(0, 2, 100),
    'Annotator_5': np.random.randint(0, 2, 100)
})
annotations_5_annotators['human_distribution'] = [get_normalized_human_distribution(annotations_5_annotators.iloc[i], 2) for i in range(annotations_5_annotators.shape[0])]

# Simulate predictions by two models (Model A and Model B) for 5 annotators
model_A_predictions_5 = np.random.uniform(0.3, 0.7, 100)  
model_B_predictions_5 = np.random.uniform(0.2, 0.8, 100)  
annotations_5_annotators["MODEL_A"] = model_A_predictions_5
annotations_5_annotators["MODEL_B"] = model_B_predictions_5
annotations_5_annotators["MODEL_A"] = annotations_5_annotators["MODEL_A"].apply(lambda x: [1-x, x] if x <= 0.5 else [x, 1-x])
annotations_5_annotators["MODEL_B"] = annotations_5_annotators["MODEL_B"].apply(lambda x: [1-x, x] if x <= 0.5 else [x, 1-x])



# to each dataset add the entropy of each item and the human distribution
annotations_3_annotators['human entropy'] = annotations_3_annotators.human_distribution.apply(lambda x: normalized_entropy(x))

annotations_5_annotators['human entropy'] = annotations_5_annotators.human_distribution.apply(lambda x: normalized_entropy(x))
annotations_3_annotators['model_A_entropy'] = annotations_3_annotators.MODEL_A.apply(lambda x: normalized_entropy(x))
annotations_3_annotators['model_B_entropy'] = annotations_3_annotators.MODEL_B.apply(lambda x: normalized_entropy(x))
print(tabulate(annotations_3_annotators.head(), headers='keys', tablefmt='pretty'))


# In[ ]:





# The dataframes should contain all information that are needed to calculate different metrics. Calculate some metrics of your choice for the datasets and models and compare the performance of the two models. What do you observe?

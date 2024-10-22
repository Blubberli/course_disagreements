#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import cohen_kappa_score
from krippendorff import alpha
from pingouin import intraclass_corr
from scipy.stats import entropy
import pandas as pd
from tabulate import tabulate


# ## Assignment 1: Annotation
# 
# This notebook implements the annotation metrics discussed in the first part of the tutorial and looks at some example data.

# Let us create the example data we looked at in the tutorial. We have one dataset with two annotators who annotated three categories (three emotions). We have another dataset with three annotators who annotated the same three categories. Finally, we use the dataset with three annotators but treat the categories as continuous values, as if they are rated for, for instance something like emotion intensity on a scale from 1 to 3. We can check if the values are in line with the values on the slides.

# In[ ]:


a1 = [0, 1, 1, 2, 1]
a2 = [0, 2, 1, 2, 2]
a3 = [0, 1, 1, 1, 2]

# first compute the cohens kappa between a1 and a2
kappa = cohen_kappa_score(a1, a2)
print(f"Cohen's Kappa between a1 and a2: {kappa}")

# second, we compute the krippendorff's alpha between a1, a2, and a3
alpha = alpha([a1, a2, a3], level_of_measurement='nominal')
print(f"Krippendorff's Alpha between a1, a2, and a3: {alpha}")

# finally, we compute the ICC2k between a1, a2, and a3
# first we need to reshape the data, such that we have a dataframe with three columns: annotators, items and ratings
data = pd.DataFrame({'annotator': ['a1']*5 + ['a2']*5 + ['a3']*5,
                     'item': [1, 2, 3, 4, 5]*3,
                     'rating': a1 + a2 + a3})
icc = intraclass_corr(data, targets='item', raters='annotator', ratings='rating', nan_policy='omit')
print("ICC values:")
print(tabulate(icc, headers='keys', tablefmt='fancy_grid'))


# Q1: What is the cohen's kappa between annotator a2 and a3?
# 
# Q2: Why is the ICC value higher than Cohen's Kappa and Krippendorff's Alpha?

# Consider the following example of annotated data. Two annotators had to classify an image into whether it contains a dog (0) or a cat(1). Look at the result for the percentage of agreement and Cohen's Kappa. 

# In[ ]:


import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generating data
# Rater 1 has the skewed distribution: 95% cats, 5% dogs
rater1 = np.array([1] * 95 + [0] * 5)

# Rater 2 also rates based on a skewed perception
deviation = np.random.rand(100) < 0.1  # 10% deviation
rater2 = np.where(deviation, 1 - rater1, rater1)  # Flipping the label in case of a deviation

# Calculate the percentage of agreement
agreement_percentage = np.mean(rater1 == rater2) * 100

kappa = cohen_kappa_score(rater1, rater2)

# Create a DataFrame for visualization
data = pd.DataFrame({'Animal': ['Cat']*95 + ['Dog']*5, 'Rater 1': rater1, 'Rater 2': rater2})
print("Sample of the ratings:")
print(data.sample(10))  # Show a random sample of the data

print(f"\nPercentage of Agreement: {agreement_percentage:.2f}%")
print(f"Cohen's Kappa Score: {kappa:.2f}")

# Simple analysis of category counts
category_counts = data[['Rater 1', 'Rater 2']].apply(pd.Series.value_counts)
print("\nCategory counts per rater:")
print(category_counts)


# Q3: Why is the percentage of agreement so high, even though the Cohen's Kappa is low?

# In[ ]:


high_agreement = [0.9, 0.05, 0.05]
moderate_agreement = [0.5, 0.3, 0.2]
equal_agreement = [0.33, 0.33, 0.33]
one_dominant =  [0.7, 0.2, 0.1]
one_majority = [0.8, 0.1, 0.1]


# In[ ]:


def normalized_entropy(prob_dist):
    # Calculate entropy with base 2 (in bits)
    ent = entropy(prob_dist, base=2)
    # Max entropy occurs when all categories are equally likely
    max_ent = np.log2(len(prob_dist))
    # Normalize the entropy (divide by max possible entropy) take into account that entropy is 0 when all are the same
    if max_ent == 0:
        norm_entropy = 0
    else:
        norm_entropy = ent / max_ent
    return ent, norm_entropy


# Q4: Calculate the entropy and normalized entropy for the different distributions and sort them from low to high entropy. How is it related to the agreement between the raters?

# In[ ]:


# High agreement: most raters agree
data_high = pd.DataFrame({
    'Item': [1, 2, 3, 4, 5],
    'Rater 1': ['A', 'A', 'B', 'B', 'C'],
    'Rater 2': ['A', 'C', 'B', 'B', 'C'],
    'Rater 3': ['A', 'A', 'B', 'B', 'C']
})

# Moderate agreement: raters have some disagreement
data_moderate = pd.DataFrame({
    'Item': [1, 2, 3, 4, 5],
    'Rater 1': ['A', 'B', 'B', 'A', 'C'],
    'Rater 2': ['A', 'C', 'B', 'A', 'C'],
    'Rater 3': ['A', 'B', 'A', 'A', 'B']
})

# Low agreement: raters have little agreement
data_low = pd.DataFrame({
    'Item': [1, 2, 3, 4, 5],
    'Rater 1': ['A', 'C', 'B', 'A', 'C'],
    'Rater 2': ['B', 'C', 'A', 'B', 'A'],
    'Rater 3': ['C', 'A', 'C', 'A', 'B']
})


# In[ ]:


def calculate_entropy_for_items(input_data):
    items_entropy = []
    for _, row in input_data.iterrows():
        # Get the frequency distribution of categories for each item
        counts = row[1:].value_counts(normalize=True)
        ent, norm_ent = normalized_entropy(counts)
        items_entropy.append({'Item': row['Item'], 'Normalized Entropy': norm_ent})
    return pd.DataFrame(items_entropy)

entropy_high = calculate_entropy_for_items(data_high)
entropy_moderate = calculate_entropy_for_items(data_moderate)
entropy_low = calculate_entropy_for_items(data_low)

# Display Entropy Results for Each Scenario
print("Entropy for High Agreement Scenario:")
print(entropy_high)
print("\nEntropy for Moderate Agreement Scenario:")
print(entropy_moderate)
print("\nEntropy for Low Agreement Scenario:")
print(entropy_low)


# In[ ]:


kappa_high = cohen_kappa_score(data_high['Rater 2'], data_high['Rater 3'])
kappa_moderate = cohen_kappa_score(data_moderate['Rater 2'], data_moderate['Rater 3'])
kappa_low = cohen_kappa_score(data_low['Rater 2'], data_low['Rater 3'])

# Display Cohen's Kappa Results for Each Scenario
print("\nCohen's Kappa for High Agreement Scenario")
print(kappa_high)
print("\nCohen's Kappa for Moderate Agreement Scenario")
print(kappa_moderate)
print("\nCohen's Kappa for Low Agreement Scenario")
print(kappa_low)


# In[ ]:


from matplotlib import pyplot as plt

# Plotting Normalized Entropy for Each Scenario
scenarios = ['High Agreement', 'Moderate Agreement', 'Low Agreement']
average_entropy = [
    entropy_high['Normalized Entropy'].mean(),
    entropy_moderate['Normalized Entropy'].mean(),
    entropy_low['Normalized Entropy'].mean()
]

kappa_scenarios = [kappa_high, kappa_moderate, kappa_low]
# create a bar plot that compares a bar for entropy and for kappa for each scenario. each color represents a different metric
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(scenarios))
bar1 = ax.bar(index, average_entropy, bar_width, label='Entropy')
bar2 = ax.bar(index + bar_width, kappa_scenarios, bar_width, label="Cohen's Kappa")
# add the labels of the scenarios to the x-axis
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(scenarios)
ax.set_ylabel('Value')
ax.set_title('Entropy and Cohen\'s Kappa for Different Agreement Scenarios')
ax.legend()
plt.show()


# Q5: Compare the two metrics for the different scenarios using the plot. How do they differ in general between scenarios? How does kappa differ from entropy?

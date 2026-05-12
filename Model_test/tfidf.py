#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# In[2]:


import json
import pandas as pd
import numpy as np
import math

from sklearn.feature_extraction.text import TfidfVectorizer

get_ipython().system('pip install transformers datasets')
get_ipython().system('pip install faiss-cpu')



import faiss


# In[3]:


# Load SQuAD data
file_path = 'train.json'

def convert_squad_data_to_dataframe(file_path, cnt):
    with open(file_path, 'r') as f:
        squad_dict = json.load(f)
        data = squad_dict['data']
        rows = []
        for article in data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    qid = qa['id']
                    is_impossible = qa.get('is_impossible', False)
                    answers = qa['answers'] if not is_impossible else []
                    if answers:
                        for answer in answers:
                            rows.append({
                                'context': context,
                                'question': question,
                                'qid': qid,
                                'answer': answer['text'],
                                'answer_start': answer['answer_start'],
                                'c_id': cnt
                            })
                    else:
                        rows.append({
                            'context': context,
                            'question': question,
                            'qid': qid,
                            'answer': '',
                            'answer_start': -1,
                            'c_id': cnt
                        })
                cnt += 1
        return pd.DataFrame(rows)



# In[4]:


cnt = 0
data = convert_squad_data_to_dataframe(file_path, cnt)

# Drop duplicate contexts
documents = data[['context', 'c_id']].drop_duplicates().reset_index(drop=True)

# Create mapping for questions to their correct context IDs
questionContext = dict(zip(data['question'], data['c_id']))

# Vectorize contexts using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(documents['context'])

# Convert sparse to dense and normalize for FAISS
X_dense = X.toarray().astype(np.float32)
faiss.normalize_L2(X_dense)

# Build FAISS index
index = faiss.IndexFlatIP(X_dense.shape[1])
index.add(X_dense)

# Compute metrics
def dcg(scores):
    return sum([rel / math.log2(idx + 2) for idx, rel in enumerate(scores)])

def compute_metrics(question, questionContext, index, k, max_samples):
    total_correct = 0
    reciprocal_ranks = []
    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    mar_scores = []

    for idx, ques in enumerate(question):
        if idx >= max_samples:
            break
        query_vector = vectorizer.transform([ques]).toarray().astype(np.float32)
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, k)

        predictions = indices[0]
        actual_index = questionContext[ques]

        relevance = [1 if pred == actual_index else 0 for pred in predictions]

        if actual_index in predictions:
            total_correct += 1
            rank = list(predictions).index(actual_index)
            reciprocal_ranks.append(1.0 / (rank + 1))
            recall_scores.append(1.0)
            precision_scores.append(1.0 / (rank + 1))
        else:
            reciprocal_ranks.append(0.0)
            recall_scores.append(0.0)
            precision_scores.append(0.0)

        ideal_relevance = sorted(relevance, reverse=True)
        ndcg = dcg(relevance) / dcg(ideal_relevance) if sum(ideal_relevance) > 0 else 0.0
        ndcg_scores.append(ndcg)

        num_relevant_items = sum(relevance)
        mar_scores.append(num_relevant_items / len(relevance) if num_relevant_items > 0 else 0.0)

    num_samples = min(len(question), max_samples)

    return {
        f"Accuracy@{k}": (total_correct / num_samples) * 100,
        f"MRR@{k}": np.mean(reciprocal_ranks),
        f"NDCG@{k}": np.mean(ndcg_scores),
        f"Recall@{k}": np.mean(recall_scores),
        f"Precision@{k}": np.mean(precision_scores),
        f"MAR@{k}": np.mean(mar_scores)
    }


# In[5]:


# Evaluation

final_result = []
questions = list(questionContext.keys())




# In[6]:


for k in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 250, 300, 350, 400, 500, 600, 700, 800, 1000]:
        metrics = compute_metrics(questions, questionContext, index, k=k, max_samples=86769)
        final_result.append(metrics)
        print(f"\n🔍 Top-{k} Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


# In[ ]:





# In[ ]:


with open("final_result.json", "w") as f:
    json.dump(final_result, f)


# In[ ]:





# In[ ]:





# ### 

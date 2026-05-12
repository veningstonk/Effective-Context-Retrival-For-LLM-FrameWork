#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import json
import pandas as pd
import numpy as np
import math
import torch

from sklearn.feature_extraction.text import TfidfVectorizer

get_ipython().system('pip install transformers datasets')
get_ipython().system('pip install faiss-cpu')



import faiss


# In[ ]:


from transformers import AutoTokenizer , AutoModel
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)
model.eval().cuda()


# In[ ]:


# Load SQuAD data
file_path = '/kaggle/input/squid-dataset/train.json'

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



# In[ ]:


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


# In[ ]:


import torch.nn.functional as F
from tqdm import tqdm


# Mean Pooling Function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, dim=1) / input_mask_expanded.sum(dim=1)
    return pooled

# Encode Single Batch of Texts
def encode_text(texts):
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)
    
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)  # L2 normalize

    return sentence_embeddings.detach().cpu().numpy()


def encode_in_batches(texts, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = encode_text(batch_texts)
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# In[ ]:


texts = documents['context'].tolist()
doc_embeddings = encode_in_batches(texts, batch_size=32)
dim = doc_embeddings.shape[1]
doc_embeddings = np.array(doc_embeddings)  # shape: [num_docs, dim]


# In[ ]:


# Compute metrics
def dcg(scores):
    return sum([rel / math.log2(idx + 2) for idx, rel in enumerate(scores)])

def compute_metrics(question, questionContext, index, k=10, max_samples=2000, rerank_top_k=10):
    total_correct = 0
    reciprocal_ranks = []
    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    mar_scores = []

    for idx, ques in enumerate(tqdm(question[:max_samples], desc=f"Evaluating Top-{k}", leave=True, ncols=80)):
        if idx >= max_samples:
            break
        query_vector = vectorizer.transform([ques]).toarray().astype(np.float32)
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, k)

        predictions = indices[0]
        actual_index = questionContext[ques]


        
        top_k_embeddings = np.array([doc_embeddings[i] for i in predictions])
        query_embedding = encode_text([ques])

        # Step 3: Convert to tensor
        top_k_tensor = torch.tensor(top_k_embeddings).float().to(device)
        query_tensor = torch.tensor(query_embedding).float().to(device)

        # Step 4: Normalize and compute similarity
        top_k_tensor = F.normalize(top_k_tensor, p=2, dim=1)
        query_tensor = F.normalize(query_tensor, p=2, dim=1)
        sim_scores = torch.matmul(top_k_tensor, query_tensor.T).squeeze(1)

        # Step 5: Rerank
        reranked_indices = torch.argsort(sim_scores, descending=True).tolist()
        reranked_predictions = [predictions[i] for i in reranked_indices[:rerank_top_k]]

        
        
        # relevance = [1 if pred == actual_index else 0 for pred in predictions]

        relevance = [1 if pred == actual_index else 0 for pred in reranked_predictions]

        
        # if actual_index in predictions:
        #     total_correct += 1
        #     rank = list(predictions).index(actual_index)
        #     reciprocal_ranks.append(1.0 / (rank + 1))
        #     recall_scores.append(1.0)
        #     precision_scores.append(1.0 / (rank + 1))
        # else:
        #     reciprocal_ranks.append(0.0)
        #     recall_scores.append(0.0)
        #     precision_scores.append(0.0)

        if actual_index in reranked_predictions:
            total_correct += 1
            rank = reranked_predictions.index(actual_index)
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


# In[ ]:


# Evaluation

final_result = []
questions = list(questionContext.keys())

Q= questions


# In[ ]:


print("Tell me the value of K : ")
for r in [1,2,3,5,10,15,20,25,30,35,40,45,50, 100]:
    metrics = compute_metrics(Q, questionContext, index, k=100, max_samples=86769, rerank_top_k = r)
    final_result.append(metrics)
    print(f"\n🔍 Top-{k} Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


# In[ ]:


# for k in [100]:  # or any list
#     metrics = compute_metrics(Q, questionContext, index, k=k, max_samples=86769, rerank_top_k = 15)
#     final_result.append(metrics)
#     print(f"\n🔍 Top-{k} Evaluation Metrics:")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value:.4f}")


# In[ ]:


# for k in [100]:  # or any list
#     metrics = compute_metrics(Q, questionContext, index, k=k, max_samples=86769, rerank_top_k = 20)
#     final_result.append(metrics)
#     print(f"\n🔍 Top-{k} Evaluation Metrics:")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value:.4f}")


# In[ ]:


# for k in [100]:  # or any list
#     metrics = compute_metrics(Q, questionContext, index, k=k, max_samples=86769, rerank_top_k = 5)
#     final_result.append(metrics)
#     print(f"\n🔍 Top-{k} Evaluation Metrics:")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value:.4f}")


# In[ ]:


# for k in [100]:  # or any list
#     metrics = compute_metrics(Q, questionContext, index, k=k, max_samples=86769, rerank_top_k = 25)
#     final_result.append(metrics)
#     print(f"\n🔍 Top-{k} Evaluation Metrics:")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value:.4f}")


# In[ ]:


with open("/kaggle/working/final_result.json", "w") as f:
    json.dump(final_result, f)


# ### 

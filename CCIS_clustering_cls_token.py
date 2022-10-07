import os
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer, ElectraForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer, util

# cls pooling
def cls_embedding(model_output):
    last_hidden_states=model_output.hidden_states[-1]
    cls_vector=last_hidden_states[:,0,:]
    return cls_vector


# 학습된 모델을 불러와서 활용하기 
#model,tokenizer load
model_path = './dev-test-model'
model=ElectraForSequenceClassification.from_pretrained(model_path)
tokenizer=AutoTokenizer.from_pretrained('beomi/KcELECTRA-base')


# Corpus with example sentences
data_path='./data'
files=os.listdir(data_path)
data={"text":[], "label":[]}

for file in files:
    file_path=os.path.join(data_path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            data["text"].append(line.split('\t')[0])
            data["label"].append(line.split('\t')[1].strip())

print(data["text"][:10])
print(data["label"][:10])
print()

# corpus encoding (create embedding)
corpus = data["text"]

encoded_input=tokenizer(corpus, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output=model(**encoded_input, output_hidden_states=True)

sentence_embeddings=cls_embedding(model_output)

print(sentence_embeddings)

# k-means clustering using sklearn:
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(sentence_embeddings)
cluster_assignment = clustering_model.labels_

# results of k-means clustering
clustered_sentences = [[] for i in range(num_clusters)]
clustered_labels = [[] for i in range(num_clusters)]

for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])
    clustered_labels[cluster_id].append(data["label"][sentence_id])

#evaluate clustering accuracy

with open(f'./cls-clustering-train-5.txt', 'w', encoding='utf-8') as fw:
    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i + 1)
        fw.write(f"Cluster {i+1} \n")
        print(cluster)
        fw.write(f"{cluster}\n\n")
        print("")
    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i+1)
        fw.write(f"Cluster {i+1} \n")
        print("cluster에 포함된 sentence 개수: ", len(cluster))
        fw.write(f"cluster에 포함된 sentence 개수: {len(cluster)}\n")
        print(f"협박 데이터 {clustered_labels[i].count('020121')}개, {clustered_labels[i].count('020121')/len(cluster)*100:.4f}%")
        fw.write(f"협박 데이터 {clustered_labels[i].count('020121')}개, {clustered_labels[i].count('020121')/len(cluster)*100:.4f}%\n")
        print(f"갈취 및 공갈 데이터 {clustered_labels[i].count('02051')}개, {clustered_labels[i].count('02051') / len(cluster)*100:.4f}%")
        fw.write(f"갈취 및 공갈 데이터 {clustered_labels[i].count('02051')}개, {clustered_labels[i].count('02051') / len(cluster)*100:.4f}%\n")
        print(f"직장 내 괴롭힘 데이터 {clustered_labels[i].count('020811')}개, {clustered_labels[i].count('020811') / len(cluster)*100:.4f}%")
        fw.write(f"직장 내 괴롭힘 데이터 {clustered_labels[i].count('020811')}개, {clustered_labels[i].count('020811') / len(cluster)*100:.4f}%\n")
        print(f"기타 괴롭힘 데이터 {clustered_labels[i].count('020819')}개, {clustered_labels[i].count('020819') / len(cluster)*100:.4f}%")
        fw.write(f"기타 괴롭힘 데이터 {clustered_labels[i].count('020819')}개, {clustered_labels[i].count('020819') / len(cluster)*100:.4f}%\n")
        print(f"일반 대화 데이터 {clustered_labels[i].count('000001')}개, {clustered_labels[i].count('000001') / len(cluster)*100:.4f}%")
        fw.write(f"일반 대화 데이터 {clustered_labels[i].count('000001')}개, {clustered_labels[i].count('000001') / len(cluster)*100:.4f}%\n\n")
        print()

fw.close()





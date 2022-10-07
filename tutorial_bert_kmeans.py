from sentence_transformers import SentenceTransformer, util
from transformers import ElectraForSequenceClassification

model_path = './model-kcelectra'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['너를 죽여버릴거야.',
          '새끼 손가락 한 번 잘려볼래?.',
          '야 너 돈 있냐? 잠깐 가져와봐',
          '잠깐만 빌려줘 비싼 시계 같은데 내가 쓰자',
          '김대리 오늘 왜이렇게 못생겼어?',
          '영석씨 우리 집에 와서 주말에 청소나 해요 계약직이니까 잘리기 싫으면 잘해요',
          '우리 쟤 따돌리자. 전교 1등 재수없어',
          '여기 와서 무릎 꿇어요',
          '나 오늘 너무 행복해',
          '경치 완전 죽인다~']

corpus_embeddings = embedder.encode(corpus)

# Then, we perform k-means clustering using sklearn:
from sklearn.cluster import KMeans

num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")
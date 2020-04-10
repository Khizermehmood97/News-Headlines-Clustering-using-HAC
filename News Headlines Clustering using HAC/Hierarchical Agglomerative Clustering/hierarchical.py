# Tokenize and Stem Data
# Convert words to Vector Space using TFIDF matrix
# Calculate Cosine Similarity and generate the distance matrix
# Uses Ward method to generate an hierarchy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import ward, dendrogram, linkage, fcluster
import os
import nltk

#nltk.download('stopwords')
#nltk.download('punkt')

# removal of stop words and stemming
def tokenize_and_stemming(text_file):
    # fetching stopwords from onix list1 and list2
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text_file)
    filtered = [w for w in words if w not in stop_words]
    #print(filtered)
    stemmed = [stemmer.stem(t) for t in filtered]

    print(stemmed)
    return stemmed


def hierarchical():

    path = os.path.abspath(os.path.dirname(__file__))
    data = pd.read_csv(os.path.join(path, 'data\dataset_cleaned.txt'), names=['text'])

    
    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Performing tf*idf to convert words to Vector Space
    tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                       use_idf=True,
                                       stop_words='english',
                                       tokenizer=tokenize_and_stemming)
    

    # Fit the tfidf vector to text
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])

    # Calculating the distance using cosine similarity
    distance = 1 - cosine_similarity(tfidf_matrix)

    # Performing hierarchical  clusterings
    #linkage_matrix = ward(distance)
    linkage_matrix = linkage(distance, 'complete' )
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="top", labels=data.values)
    plt.tight_layout()
    plt.title('News Headlines Clustering using HAC Method')
    plt.savefig(os.path.join(path, 'results\hierarchical.png'))

    cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='complete')
   # clusters = fcluster(distance, 5, criterion='distance')
    cluster_label = cluster.fit_predict(distance)
    print('\nClusters Array: \n')
    print (cluster_label)


    # sample_silhouette_values = silhouette_samples(distance, cluster_label)
    # print (sample_silhouette_values)

    i=0
    cluster1 = cluster2= cluster3 = cluster4 = cluster5 = cluster6 = cluster7 = cluster8 = 0
    list1 = [0]*8
    
    while (i < len(cluster_label)):
        list1[cluster_label[i]] += 1
        i+=1

    print('\nClusters count\n')
    # print(cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8)
    print(list1)


    # scatter plot
    # plt.figure(figsize=(10, 8))  
    # plt.scatter(distance[:,0], distance[:,1], c=cluster_label, cmap='rainbow') 
    # plt.savefig(os.path.join(path, 'results\clusters.png'))


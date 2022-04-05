import pandas as pd
import gensim
from t2i import T2I
from nltk.tokenize import word_tokenize
# import nltk
# # nltk.data.path.append("F:/Anaconda/Lib/site-packages/nltk_data")
# nltk.download('punkt')
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load the queries from every dataset
queries_test = pd.read_csv('query-VS/dataset/videos/video_name_test.csv', engine='python')
queries_train = pd.read_csv('query-VS/dataset/videos/video_name_train.csv', engine='python')
queries_val = pd.read_csv('query-VS/dataset/videos/video_name_val.csv', engine='python')

# Define the vector that will contain all the queries
# queries = []

# # Adding training queries
# for idx in range(queries_train.size):
#     queries.append(queries_train.iloc[idx, 0])
#
# # Adding validation queries
# for idx in range(queries_val.size):
#     queries.append(queries_val.iloc[idx, 0])
#
# # Adding testing queries
# for idx in range(queries_test.size):
#     queries.append(queries_test.iloc[idx, 0])

# Stack train/val/test queries together
df_queries = pd.concat([queries_train, queries_val, queries_test], axis=0)
queries = df_queries.values.tolist()

# Tokenize
token = [word_tokenize(q[0]) for q in queries]

# Build the dictionary
word_index = T2I.build(token)

# Build word2vec model
model = gensim.models.Word2Vec(token, min_count=1)
print(model)
# summarize vocabulary
words = list(model.wv.key_to_index)
print(len(words))
# print(model.wv['3d'])
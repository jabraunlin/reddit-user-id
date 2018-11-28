import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, FloatType, IntegerType, ArrayType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer, Tokenizer, HashingTF, StandardScaler, Normalizer
from pyspark.ml.feature import StopWordsRemover
import re
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import pandas as pd
from nltk.util import skipgrams
from itertools import chain
from scipy.cluster import hierarchy

spark = (
    ps.sql.SparkSession.builder
    .master("local[4]")
    .appName("project1")
    .getOrCreate()
)

sc = spark.sparkContext


# Read sharded json files
df = spark.read.json("*.json")

# Find authors that have more than 400 comments so that splitting them leaves us with at least 200 comments
new_df = df.groupby('author').agg(F.count('body'))
authors = new_df.filter(new_df['count(body)'] > 400).select('author')

# Filter the original data with authors that have more than 400 comments
filtered_df = authors.join(df, ['author'], 'left')

# Split users into users and pseudo-users to compare them
df1, df2 = filtered_df.randomSplit([0.5, 0.5])

# Concatenate comments into corpora of each user's entire comment history
comments1 = df1.groupBy("author").agg(F.collect_list("body"))
join_comments_udf = udf(lambda x: ' '.join(x), StringType())
df1_join_comments = comments1.withColumn(
    'corpus', join_comments_udf(comments1['collect_list(body)']))

# Data Cleaning
# Count and Drop Links


def count_links(s):
    try:
        num_links = len(re.findall(r'\(http.+\)', s)[0].split(')('))
        return num_links
    except:
        return 0


count_links_udf = udf(count_links, IntegerType())
df_count_links1 = df1_join_comments.withColumn(
    'link_count', count_links_udf(df1_join_comments['corpus']))


def drop_links(s):
    return re.sub(r'\(http.+\)', '', s)


drop_links_udf = udf(drop_links, StringType())
df_drop_links1 = df_count_links1.withColumn('corpus', drop_links_udf(df_count_links1['corpus']))

# Tokenize words


def tokenize(s):
    s = s.lower()
    token = TweetTokenizer()
    return token.tokenize(s)


tokenize_udf = udf(tokenize, ArrayType(StringType()))
df_tokens1 = df_drop_links1.withColumn('tokens', tokenize_udf(df_drop_links1['corpus']))

# Tag parts of speech for each word


def pos_tagger(s):
    return [i[1] for i in nltk.pos_tag(s)]


pos_tagger_udf = udf(pos_tagger, ArrayType(StringType()))
df_pos_tagger1 = df_tokens1.withColumn('POS', pos_tagger_udf(df_tokens1['tokens']))

# Find POS tagging tendencies to determine sentence structure patterns


def skip_grams(s):
    grams = []
    for i in skipgrams(s, 2, 2):
        grams.append(str(i))
    return grams


skip_grams_udf = udf(skip_grams, ArrayType(StringType()))
df_skip_grams1 = df_pos_tagger1.withColumn('skip_grams', skip_grams_udf(df_pos_tagger1['POS']))

# Open file containing the most common skip grams I had previously found from analyzing a previous sample
import csv

with open('skip_grams.csv', 'r') as f:
    reader = csv.reader(f)
    com_skips = list(reader)

skips = com_skips[0]

# Filter through each user's POS skip-grams and keep them if they are in the most commonly found skip-grams


def skip_grams_filter(s):
    return [i for i in s if i in skips]


com_skips_udf = udf(skip_grams_filter, ArrayType(StringType()))
df_com_skips1 = df_skip_grams1.withColumn('com_skips', com_skips_udf(df_skip_grams1['skip_grams']))

# Create stop words feature list and add extra features
stops = stopwords.words('english')
x = [i.split("'")for i in stops]
stops = [i[0] for i in x]
stops = list(set(stops))
slang_stops = ['gonna', 'coulda', 'shoulda',
               'lotta', 'lots', 'oughta', 'gotta', 'ain', 'sorta', 'kinda', 'yeah', 'whatever', 'cuz', 'ya', 'haha', 'lol', 'eh']
puncts = ['!', ':', '...', '.', '%', '$', "'", '"', ';']
formattings = ['##', '__', '_', '    ', '*', '**']

stops.extend(slang_stops)
stops.extend(puncts)
stops.extend(formattings)
stops.extend(skips)

# Filter words with the list of stop words


def stop_words_filter(s):
    return [i for i in s if i in stops]


stop_words_udf = udf(stop_words_filter, ArrayType(StringType()))
df_stop_words1 = df_com_skips1.withColumn('stop_words', stop_words_udf(df_com_skips1['tokens']))

# Concatenate each user's list of function words and skip grams to a single array


def concat(type):
    def concat_(*args):
        return list(chain.from_iterable((arg if arg else [] for arg in args)))
    return udf(concat_, ArrayType(type))


concat_arrays_udf = concat(StringType())
df_all_words1 = df_stop_words1.select("author", concat_arrays_udf("stop_words", "com_skips"))

# Count Vectorize the combined function word and skip gram array

hashingTF = HashingTF(
    numFeatures=285, inputCol='concat_(stop_words, com_skips)', outputCol='features')
tf1 = hashingTF.transform(df_all_words1)

# Normalize the counts so that they are a percentage of total counts of the features

tf_norm1 = Normalizer(inputCol="features", outputCol="features_norm", p=1).transform(tf1)

# Standardize the vector based on average use of each feature among all users
stdscaler = StandardScaler(inputCol='features_norm', outputCol='scaled', withMean=True)
scale_fit1 = stdscaler.fit(tf_norm1)
scaled1 = scale_fit1.transform(tf_norm1)


# Do all of the above for subset #2

comments2 = df2.groupBy("author").agg(F.collect_list("body"))
join_comments_udf = udf(lambda x: ' '.join(x), StringType())
df2_join_comments = comments2.withColumn(
    'corpus', join_comments_udf(comments2['collect_list(body)']))

df_count_links2 = df2_join_comments.withColumn(
    'link_count', count_links_udf(df2_join_comments['corpus']))

df_drop_links2 = df_count_links2.withColumn('corpus', drop_links_udf(df_count_links2['corpus']))

df_tokens2 = df_drop_links2.withColumn('tokens', tokenize_udf(df_drop_links2['corpus']))

pos_tagger_udf = udf(pos_tagger, ArrayType(StringType()))
df_pos_tagger2 = df_tokens2.withColumn('POS', pos_tagger_udf(df_tokens2['tokens']))

skip_grams_udf = udf(skip_grams, ArrayType(StringType()))
df_skip_grams2 = df_pos_tagger2.withColumn('skip_grams', skip_grams_udf(df_pos_tagger2['POS']))

com_skips_udf = udf(skip_grams_filter, ArrayType(StringType()))
df_com_skips2 = df_skip_grams2.withColumn('com_skips', com_skips_udf(df_skip_grams2['skip_grams']))

df_stop_words2 = df_com_skips2.withColumn('stop_words', stop_words_udf(df_com_skips2['tokens']))

df_all_words2 = df_stop_words2.select("author", concat_arrays_udf("stop_words", "com_skips"))

tf2 = hashingTF.transform(df_all_words2)

tf_norm2 = Normalizer(inputCol="features", outputCol="features_norm", p=1).transform(tf2)

scaled2 = scale_fit1.transform(tf_norm2)

# Calculate the cosine similarity for each author in subset 1 against every author in subset 2
sims1 = scaled1.select('author', 'scaled')
sims2 = scaled2.select('author', 'scaled')
similarities = {}
for i in sims1.rdd.collect():
    similarity = {}
    auth1, vec1 = i[0], i[1]
    for j in sims2.rdd.collect():
        auth2, vec2 = j[0], j[1]
        cos = vec1.dot(vec2) / (vec2.norm(2)*vec1.norm(2))
        similarity[auth2] = cos
    similarities[auth1] = similarity

pdf = pd.DataFrame(similarities)

# split the cosines of authors who match with the authors who don't match
cols = pdf.columns
mask = []
for i in pdf:
    mask.append(i == pdf.index)
mask = np.array(mask)
mask = mask.T

matches = pdf.values[mask]
non_matches = pdf.values[~mask]

# Calculate accuracy of the model
non_mas = non_matches.reshape(len(matches), -1)
non_mas_max = np.max(non_mas, axis=1)
np.sum(matches > non_mas_max) / len(matches)

# Read saved matches and non-matches
with open('nonmatches.csv', 'r') as f:
    reader = csv.reader(f)
    nonma_list = list(reader)

with open('matches.csv', 'r') as f:
    reader = csv.reader(f)
    match_list = list(reader)

match_list = [[float(x) for x in i] for i in match_list]
match_list = match_list[0]

nonma_list = [[float(x) for x in i] for i in nonma_list]
nonma_list = nonma_list[0]

# Calculate cosine threshold and power for a given alpha level
n = norm.ppf(0.9999) * np.std(nonma_list) - np.mean(nonma_list)

1 - norm.cdf(n, np.mean(match_list), np.std(match_list))


# Plotting
# Dendogram
sparkdf = scaled1.select('author', 'scaled')
pandaDF = sparkdf.toPandas()
series = pandaDF['scaled'].apply(lambda x: np.array(x.toArray())).as_matrix().reshape(-1, 1)
features = np.apply_along_axis(lambda x: x[0], 1, series)
df = pd.DataFrame(features, index=pandaDF['author'])

threshold = 0.405
Z = hierarchy.linkage(df, 'single', metric="cosine")
hierarchy.set_link_color_palette(None)

fig, axes = plt.subplots(1, 1, figsize=(15, 7))
hierarchy.dendrogram(Z, ax=axes, color_threshold=threshold, labels=df.index)
axes.axhline(y=0.405, color='r', linestyle='-', label='threshold')
axes.set_ylabel('1 - Cosine')
axes.set_title('Hierarchical Clustering')
plt.tight_layout()
plt.legend()


# Matches and non-matches hist

plt.hist(matches, label='matches')
plt.hist(non_matches, label='non-matches')
plt.xlabel('Cosine Similarity')
plt.legend()
plt.savefig('match_distro.png')


# Probability Density
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(x, norm.pdf(x, np.mean(nonma_list), np.std(nonma_list)), label='non-matches')
ax.fill_between(x, 0, norm.pdf(x, np.mean(nonma_list), np.std(nonma_list)), alpha=0.5)
ax.plot(x, norm.pdf(x, np.mean(match_list), np.std(match_list)), label='matches')
ax.fill_between(x, 0, norm.pdf(x, np.mean(match_list), np.std(match_list)), alpha=0.5)
ax.set_title('Probability Density')
ax.set_xlabel('Cosine Similarity')
ax.legend()
plt.savefig('prob_density.png')

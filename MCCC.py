import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, FloatType, IntegerType, ArrayType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer, Tokenizer, HashingTF, StandardScaler
from pyspark.ml.feature import StopWordsRemover
import re
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import pandas as pd

spark = (
    ps.sql.SparkSession.builder
    .master("local[4]")
    .appName("project")
    .getOrCreate()
)

sc = spark.sparkContext

# Read sharded json files
df = spark.read.json("*.json")

df_users = df1.groupby('author').agg(
    F.count('body')).orderBy('count(body)', ascending=False)
comments = df.groupBy("author").agg(F.collect_list("body"))
join_comments_udf = udf(lambda x: ' '.join(x), StringType())
df_join_comments = comments.withColumn('corpus', join_comments_udf(comments['collect_list(body)']))

# Clean up Reddit formatting techniques from raw comments
# Count and drop hyperlinks from comments


def count_links(s):
    num_links = len(re.findall(r'\(http.+\)', s)[0].split(')('))
    return num_links


count_links_udf = udf(count_links, IntegerType())
df_count_links = df_join_comments.withColumn(
    'link_count', count_links_udf(df_join_comments['corpus']))


def drop_links(s):
    return re.sub(r'\(http.+\)', '', s)


drop_links_udf = udf(drop_links, StringType())
df_drop_links = df_count_links.withColumn('corpus', drop_links_udf(df_count_links['corpus']))


# Count and drop bold formatting

def count_bold(s):
    x = 0
    y = 0
    try:
        x = len(re.findall(r'**.+**', s)[0].split(')('))
    except:
        x = 0
    try:
        y = len(re.findall(r'__.+__', s)[0].split(')('))
    except:
        y = 0
    return x+y


count_bold_udf = udf(count_bold, IntegerType())
df_count_bold = df_drop_links.withColumn('bold_count', count_bold_udf(df_drop_links['corpus']))


def drop_bold(s):
    try:
        return re.sub(r'**', '', s)
    except:
        try:
            return re.sub(r'__', '', s)
        except:
            return s


drop_bold_udf = udf(drop_bold, StringType())
df_drop_bold = df_count_bold.withColumn('corpus', drop_bold_udf(df_count_bold['corpus']))

# Count and drop italics formatting


def count_italics(s):
    x = 0
    y = 0
    try:
        x = len(re.findall(r'*', s)[0].split('*'))
    except:
        x = 0
    try:
        y = len(re.findall(r'_', s)[0].split('_'))
    except:
        y = 0
    return x+y


count_italics_udf = udf(count_italics, IntegerType())
df_count_italics = df_drop_bold.withColumn(
    'italics_count', count_italics_udf(df_drop_bold['corpus']))


def drop_italics(s):
    try:
        return re.sub(r'*', '', s)
    except:
        try:
            return re.sub(r'_', '', s)
        except:
            return s


drop_italics_udf = udf(drop_italics, StringType())
df_drop_italics = df_count_italics.withColumn(
    'corpus', drop_italics_udf(df_count_italics['corpus']))

# Count and drop block quote formatting


def count_blocks(s):
    try:
        return len(re.findall(r'    ', s)[0].split('    '))
    except:
        return 0


count_blocks_udf = udf(count_blocks, IntegerType())
df_count_blocks = df_drop_italics.withColumn(
    'block_count', count_blocks_udf(df_drop_italics['corpus']))


def drop_blocks(s):
    try:
        return re.sub(r'    ', '', s)
    except:
        return s


drop_blocks_udf = udf(drop_blocks, StringType())
df_drop_blocks = df_count_blocks.withColumn(
    'corpus', drop_blocks_udf(df_count_blocks['corpus']))

# Count and drop headline formatting


def count_headlines(s):
    try:
        return len(re.findall(r'##', s)[0].split('##'))
    except:
        return 0


count_headlines_udf = udf(count_headlines, IntegerType())
df_count_headlines = df_drop_blocks.withColumn(
    'headline_count', count_headlines_udf(df_drop_blocks['corpus']))


def drop_headlines(s):
    try:
        return re.sub(r'##', '', s)
    except:
        return s


drop_headlines_udf = udf(drop_headlines, StringType())
df_drop_headlines = df_count_headlines.withColumn(
    'corpus', drop_headlines_udf(df_count_headlines['corpus']))


# Count and drop newline formatting

def count_newlines(s):
    try:
        return len(re.findall(r'\n\n', s)[0].split('\n\n'))
    except:
        return 0


count_newlines_udf = udf(count_newlines, IntegerType())
df_count_newlines = df_drop_headlines.withColumn(
    'newline_count', count_newlines_udf(df_drop_headlines['corpus']))


def drop_newlines(s):
    try:
        return re.sub(r'\n', '', s)
    except:
        return s


drop_newlines_udf = udf(drop_newlines, StringType())
df_drop_newlines = df_count_newlines.withColumn(
    'corpus', drop_newlines_udf(df_count_newlines['corpus']))

# Tokenize words using nltk's tweet tokenizer, which also splits words from formatting and punctutation


def tokenize(s):
    s = s.lower()
    token = TweetTokenizer()
    return token.tokenize(s)


tokenize_udf = udf(tokenize, ArrayType(StringType()))
df_tokens = df_drop_newlines.withColumn('tokens', tokenize_udf(df_drop_newlines['corpus']))


def drop_punct(s):
    return re.sub(r'[^0-9a-zA-Z]', '', s)


drop_punct_udf = udf(drop_punct, StringType())
df_drop_punct = df_drop_newlines.withColumn('corpus1', df_drop_newlines['corpus'])

# Tokenize words, create new df called tokens
tokenizer = Tokenizer(inputCol="corpus1", outputCol="words")
tokens = tokenizer.transform(df_drop_punct)
# Drop stop words, create new df called words


# Mendenhall's Characteristic Curves of Composition
def word_length(words):
    return [len(word) for word in words]


word_length_udf = udf(word_length, ArrayType(IntegerType()))
word_length_df = tokens.withColumn('word_lengths', word_length_udf(tokens['words']))
total_words_udf = udf(lambda x: len(x), IntegerType())
total_words_df = word_length_df.withColumn('total_words', total_words_udf(word_length_df['words']))
plot_df = total_words_df.orderBy(('total_words'), ascending=False).limit(10)


# plot the Curves of Composition
MCCC1 = total_words_df1.select('author', 'word_lengths', 'total_words')
word_freqs1 = {}
for i in MCCC1.rdd.collect():
    x = []
    y = []
    for k, v in (dict(nltk.FreqDist(i[1]))).items():
        x.append(k)
        y.append(v / i[2])
    idx = np.argsort(x)[1:12]
    z = np.array(y)[idx]
    plt.plot(range(1, 12), z)
    word_freqs1[i[0]] = z


MCCC2 = total_words_df2.select('author', 'word_lengths', 'total_words')
word_freqs2 = {}
for i in MCCC2.rdd.collect():
    x = []
    y = []
    for k, v in (dict(nltk.FreqDist(i[1]))).items():
        x.append(k)
        y.append(v / i[2])
    idx = np.argsort(x)[1:12]
    z = np.array(y)[idx]
    plt.plot(range(1, 12), z)
    word_freqs2[i[0]] = z


# find RMSE for each of the users
word_lengths = {}
for key, value in word_freqs1.items():
    error_values = {}
    for k, v in word_freqs2.items():
        rmse = np.mean(np.sqrt((v-value)**2))
        error_values[k] = rmse
    word_lengths[key] = error_values


# Plotting
df = pd.DataFrame(word_lengths)
ax = df.plot(kind='bar', figsize=(16, 8), title='RMSE of User Word Length Choices')
fig = ax.get_figure()

fig, ax = plt.subplots(2, 1, figsize=(10, 12))
for key, value in word_freqs1.items():
    ax[0].plot(range(1, 12), value, label=key)
ax[0].set_ylabel('% of Total Words')
ax[0].set_xlabel('Word Length')
ax[0].set_title('Word Choice Frequencies: Subset 1')
ax[0].legend()
for key, value in word_freqs2.items():
    ax[1].plot(range(1, 12), value, label=key)
ax[1].set_ylabel('% of Total Words')
ax[1].set_xlabel('Word Length')
ax[1].set_title('Word Choice Frequencies: Subset 2')
ax[1].legend()

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

# ssh -NfL 48888:localhost:48888 mpg_website

# If you need to put a local file onto worker nodes:
# hadoop fs -put filename /user/hadoop/file.csv
spark = (
    ps.sql.SparkSession.builder
    .master("local[4]")
    .appName("project")
    .getOrCreate()
)

sc = spark.sparkContext


df = spark.read.json("sample.json")

df1 = df[['author', 'body']]
df_users = df1.groupby('author').agg(
    F.count('body')).orderBy('count(body)', ascending=False)
comments = df.groupBy("author").agg(F.collect_list("body"))
join_comments_udf = udf(lambda x: ' '.join(x), StringType())
df_join_comments = comments.withColumn('corpus', join_comments_udf(comments['collect_list(body)']))
# Count and drop hyperlinks


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
# stopwords = StopWordsRemover(inputCol="words", outputCol="true_words")
# words = stopwords.transform(tokens)


# Mendenhall's Characteristic Curves of Composition
def word_length(words):
    return [len(word) for word in words]


word_length_udf = udf(word_length, ArrayType(IntegerType()))
word_length_df = tokens.withColumn('word_lengths', word_length_udf(tokens['words']))
total_words_udf = udf(lambda x: len(x), IntegerType())
total_words_df = word_length_df.withColumn('total_words', total_words_udf(word_length_df['words']))
plot_df = total_words_df.orderBy(('total_words'), ascending=False).limit(10)


# plot the Curves of Composition
MCCC = plot_df.select('author', 'word_lengths', 'total_words')
word_freqs = {}
for i in MCCC.rdd.collect():
    x = []
    y = []
    for k, v in (dict(nltk.FreqDist(i[1]))).items():
        x.append(k)
        y.append(v / i[2])
    idx = np.argsort(x)[:15]
    z = np.array(y)[idx]
    plt.plot(range(15), z)
    word_freqs[i[0]] = z

# find RMSE for each of the users
word_lengths = {}
for key, value in word_freqs1.items():
    error_values = {}
    for k, v in word_freqs2.items():
        rmse = np.mean(np.sqrt((v-value)**2))
        error_values[k] = rmse
    word_lengths[key] = error_values

# Create a list of function words
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


def stop_words_filter(s):
    return [i for i in s if i in stops]


stop_words_udf = udf(stop_words_filter, ArrayType(StringType()))
df_stop_words = df_tokens.withColumn('stop_words', stop_words_udf(df_tokens['tokens']))


hashingTF = HashingTF(numFeatures=153, inputCol='stop_words', outputCol='features')
tf = hashingTF.transform(df_stop_words)

tf_norm = Normalizer(inputCol="features", outputCol="features_norm", p=1).transform(tf)

stdscaler = StandardScaler(inputCol='features_norm', outputCol='scaled', withMean=True)
scale_fit = stdscaler.fit(tf_norm)
scaled = scale_fit.transform(tf_norm)


# NEW IPYNB

import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
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


spark = (
    ps.sql.SparkSession.builder
    .master("local[4]")
    .appName("project")
    .getOrCreate()
)

sc = spark.sparkContext


df = spark.read.json("42_users.json")


df1, df2 = df.randomSplit([0.5, 0.5])

df2.groupby('author').agg(F.count('body')).show()

df1.groupby('author').agg(F.count('body')).show()

comments1 = df1.groupBy("author").agg(F.collect_list("body"))
join_comments_udf = udf(lambda x: ' '.join(x), StringType())
df1_join_comments = comments1.withColumn(
    'corpus', join_comments_udf(comments1['collect_list(body)']))


def count_links(s):
    try:
        num_links = len(re.findall(r'\(http.+\)', s)[0].split(')('))
        return num_links
    except:
        return 0


count_links_udf = udf(count_links, IntegerType())
df_count_links1 = df1_join_comments.withColumn(
    'link_count', count_links_udf(df1_join_comments['corpus']))
df_count_links1.show(3)


def drop_links(s):
    return re.sub(r'\(http.+\)', '', s)


drop_links_udf = udf(drop_links, StringType())
df_drop_links1 = df_count_links1.withColumn('corpus', drop_links_udf(df_count_links1['corpus']))


def tokenize(s):
    s = s.lower()
    token = TweetTokenizer()
    return token.tokenize(s)


tokenize_udf = udf(tokenize, ArrayType(StringType()))
df_tokens1 = df_drop_links1.withColumn('tokens', tokenize_udf(df_drop_links1['corpus']))


def find_words(s):
    return [i for i in s if i.isalpha()]


find_words_udf = udf(find_words, ArrayType(StringType()))
df_find_words1 = df_tokens1.withColumn('words', find_words_udf(df_tokens1['tokens']))


def word_length(words):
    return [len(word) for word in words]


word_length_udf = udf(word_length, ArrayType(IntegerType()))
word_length_df1 = df_find_words1.withColumn(
    'word_lengths', word_length_udf(df_find_words1['words']))
total_words_udf = udf(lambda x: len(x), IntegerType())
total_words_df1 = word_length_df1.withColumn(
    'total_words', total_words_udf(word_length_df1['words']))


# MCCC1 = total_words_df1.select('author', 'word_lengths', 'total_words')
# word_freqs1 = {}
# for i in MCCC1.rdd.collect():
#     x = []
#     y = []
#     for k, v in (dict(nltk.FreqDist(i[1]))).items():
#         x.append(k)
#         y.append(v / i[2])
#     idx = np.argsort(x)[1:12]
#     z = np.array(y)[idx]
#     plt.plot(range(1,12), z)
#     word_freqs1[i[0]] = z

comments2 = df2.groupBy("author").agg(F.collect_list("body"))
join_comments_udf = udf(lambda x: ' '.join(x), StringType())
df2_join_comments = comments2.withColumn(
    'corpus', join_comments_udf(comments2['collect_list(body)']))


def count_links(s):
    try:
        num_links = len(re.findall(r'\(http.+\)', s)[0].split(')('))
        return num_links
    except:
        return 0


count_links_udf = udf(count_links, IntegerType())
df_count_links2 = df2_join_comments.withColumn(
    'link_count', count_links_udf(df2_join_comments['corpus']))


def drop_links(s):
    return re.sub(r'\(http.+\)', '', s)


drop_links_udf = udf(drop_links, StringType())
df_drop_links2 = df_count_links2.withColumn('corpus', drop_links_udf(df_count_links2['corpus']))


def tokenize(s):
    s = s.lower()
    token = TweetTokenizer()
    return token.tokenize(s)


tokenize_udf = udf(tokenize, ArrayType(StringType()))
df_tokens2 = df_drop_links2.withColumn('tokens', tokenize_udf(df_drop_links2['corpus']))


def find_words(s):
    return [i for i in s if i.isalpha()]


find_words_udf = udf(find_words, ArrayType(StringType()))
df_find_words2 = df_tokens2.withColumn('words', find_words_udf(df_tokens2['tokens']))


def word_length(words):
    return [len(word) for word in words]


word_length_udf = udf(word_length, ArrayType(IntegerType()))
word_length_df2 = df_find_words2.withColumn(
    'word_lengths', word_length_udf(df_find_words2['words']))
total_words_udf = udf(lambda x: len(x), IntegerType())
total_words_df2 = word_length_df2.withColumn(
    'total_words', total_words_udf(word_length_df2['words']))

# MCCC2 = total_words_df2.select('author', 'word_lengths', 'total_words')
# word_freqs2 = {}
# for i in MCCC2.rdd.collect():
#     x = []
#     y = []
#     for k, v in (dict(nltk.FreqDist(i[1]))).items():
#         x.append(k)
#         y.append(v / i[2])
#     idx = np.argsort(x)[1:12]
#     z = np.array(y)[idx]
#     plt.plot(range(1,12), z)
#     word_freqs2[i[0]] = z


# #find RMSE for each of the users
# word_lengths = {}
# for key, value in word_freqs1.items():
#     error_values = {}
#     for k, v in word_freqs2.items():
#         rmse = np.mean(np.sqrt((v-value)**2))
#         error_values[k] = rmse
#     word_lengths[key] = error_values

# df = pd.DataFrame(word_lengths)
# ax = df.plot(kind='bar', figsize=(16,8), title='RMSE of User Word Length Choices')
# fig = ax.get_figure()
# fig.savefig('word_length_errors.png')

# fig, ax = plt.subplots(2,1,figsize=(10,12))
# for key, value in word_freqs1.items():
#     ax[0].plot(range(1,12), value, label=key)
# ax[0].set_ylabel('% of Total Words')
# ax[0].set_xlabel('Word Length')
# ax[0].set_title('Word Choice Frequencies: Subset 1')
# ax[0].legend()
# for key, value in word_freqs2.items():
#     ax[1].plot(range(1,12), value, label=key)
# ax[1].set_ylabel('% of Total Words')
# ax[1].set_xlabel('Word Length')
# ax[1].set_title('Word Choice Frequencies: Subset 2')
# ax[1].legend()
# fig.savefig('word_lengths.png', bbox_inches='tight')

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
len(stops)


def stop_words_filter(s):
    return [i for i in s if i in stops]


stop_words_udf = udf(stop_words_filter, ArrayType(StringType()))
df_stop_words1 = total_words_df1.withColumn('stop_words', stop_words_udf(total_words_df1['tokens']))

hashingTF = HashingTF(numFeatures=179, inputCol='stop_words', outputCol='features')
tf1 = hashingTF.transform(df_stop_words1)

tf_norm1 = Normalizer(inputCol="features", outputCol="features_norm", p=1).transform(tf1)

stdscaler = StandardScaler(inputCol='features_norm', outputCol='scaled', withMean=True)
scale_fit1 = stdscaler.fit(tf_norm1)
scaled1 = scale_fit1.transform(tf_norm1)

stop_words_udf = udf(stop_words_filter, ArrayType(StringType()))
df_stop_words2 = total_words_df2.withColumn('stop_words', stop_words_udf(total_words_df2['tokens']))

hashingTF = HashingTF(numFeatures=179, inputCol='stop_words', outputCol='features')
tf2 = hashingTF.transform(df_stop_words2)

tf_norm2 = Normalizer(inputCol="features", outputCol="features_norm", p=1).transform(tf2)

stdscaler = StandardScaler(inputCol='features_norm', outputCol='scaled', withMean=True)
scale_fit2 = stdscaler.fit(tf_norm2)
scaled2 = scale_fit2.transform(tf_norm2)


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

cols = pdf.columns
mask = []
for i in pdf:
    mask.append(i == pdf.index)
mask = np.array(mask)
mask = mask.T

matches = pdf.values[mask]

non_matches = pdf.values[~mask]

sample = pd.read_json('sample.json', lines=True)

non_mas = non_matches.reshape(41, -1)

non_mas_max = np.max(non_mas, axis=1)

np.sum(matches > non_mas_max) / len(matches)

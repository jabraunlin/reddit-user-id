import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, FloatType, IntegerType, ArrayType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.ml.feature import StopWordsRemover
import re
import matplotlib.pyplot as plt
import nltk
import numpy as np

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
df_join_comments = df_join_comments.withColumn(
    'link_count', count_links_udf(df_join_comments['corpus']))


def drop_links(s):
    return re.sub(r'\(http.+\)', '', s)


drop_links_udf = udf(drop_links, StringType())
df_join_comments = df_join_comments.withColumn('corpus', drop_links_udf(df_join_comments['corpus']))


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
    'newline_count', count_newlines_udf(df_count_newlines['corpus']))


def drop_newlines(s):
    try:
        return re.sub(r'\n', '', s)
    except:
        return s


drop_newlines_udf = udf(drop_newlines, StringType())
df_drop_newlines = df_count_newlines.withColumn(
    'corpus', drop_newlines_udf(df_count_newlines['corpus']))


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
total_words_df = word_length_df.withColumn('total_words', word_length_udf(word_length_df['words']))
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

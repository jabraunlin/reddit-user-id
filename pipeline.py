import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, FloatType, IntegerType, ArrayType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.ml.feature import StopWordsRemover

# If you need to put a local file onto worker nodes:
# hadoop fs -put filename /user/hadoop/file.csv
spark = (
    ps.sql.SparkSession.builder
    .master("local[4]")
    .appName("lecture")
    .getOrCreate()
)

sc = spark.sparkContext


df = spark.read.json("sample.json")

df1 = df[['author', 'body', 'subreddit']]
df_users = df1.groupby('author', 'subreddit').agg(
    F.count('body')).orderBy('count(body)', ascending=False)
comments = df.groupBy("author", 'subreddit').agg(F.collect_list("body"))
join_comments_udf = udf(lambda x: ' '.join(x), StringType())
df_join_comments = comments.withColumn('corpus', join_comments_udf(comments['collect_list(body)']))
# Tokenize words, create new df called tokens
tokenizer = Tokenizer(inputCol="corpus", outputCol="words")
tokens = tokenizer.transform(df_join_comments)
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

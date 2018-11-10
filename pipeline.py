import pyspark as ps
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, FloatType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.ml.feature import StopWordsRemover


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
stopwords = StopWordsRemover(inputCol="words", outputCol="true_words")
words = stopwords.transform(tokens)

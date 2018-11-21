#!/bin/bash
source ~/.bashrc
export SPARK_HOME=/usr/lib/spark
export PYTHONPATH=${SPARK_HOME}/python:$PYTHONPATH
export PYSPARK_PYTHON=$HOME/anaconda/bin/python
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --NotebookApp.ip='0.0.0.0' --NotebookApp.port=48888 --NotebookApp.iopub_data_rate_limit=1000000000"

${SPARK_HOME}/bin/pyspark \
	--master yarn \
        --deploy-mode client \
	--executor-memory 26G \
        --executor-cores 4 \
	--driver-memory 26G \
        --driver-cores 4 \
	--packages com.databricks:spark-csv_2.11:1.5.0 \
	--packages com.amazonaws:aws-java-sdk-pom:1.10.34 \
	--packages org.apache.hadoop:hadoop-aws:2.7.3

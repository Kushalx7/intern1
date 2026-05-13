FROM bitnami/spark:3.4.1

USER root

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Spark-Kafka Connector JARs 
# (Necessary for Spark to talk to Kafka)
ADD https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.4.1/spark-sql-kafka-0-10_2.12-3.4.1.jar /opt/bitnami/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar /opt/bitnami/spark/jars/
ADD https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.0/kafka-clients-3.4.0.jar /opt/bitnami/spark/jars/

# Set working directory
WORKDIR /app
COPY app/ /app/app/

# Environment variables
ENV PYTHONPATH=/app
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

USER 1001

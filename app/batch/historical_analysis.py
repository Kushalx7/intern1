from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, date_format, month, desc

spark = SparkSession.builder \
    .appName("HistoricalStockAnalysis") \
    .enableHiveSupport() \
    .getOrCreate()

df = spark.read.parquet("hdfs:///stock_data/historical/")

daily_avg = df.groupBy("symbol", "date").agg(avg("close").alias("daily_avg_close"))
monthly_trends = df.withColumn("month", month("date")).groupBy("symbol", "month") \
    .agg(avg("close").alias("monthly_avg_close"))

top_gainers = df.groupBy("symbol").agg((avg("close") - avg("open")).alias("gain")) \
    .orderBy(desc("gain"))

daily_avg.write.mode("overwrite").parquet("hdfs:///analytics/daily_avg/")
monthly_trends.write.mode("overwrite").parquet("hdfs:///analytics/monthly_trends/")
top_gainers.write.mode("overwrite").parquet("hdfs:///analytics/top_gainers/")

print("Batch analysis completed")

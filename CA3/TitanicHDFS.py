# =============================================================================
# DMLS - Computer Assignment 3
# Question 3: Titanic Data Analysis with HDFS
# File: TitanicHDFS.py
# 
# Based on Apache Spark Official Examples:
# https://github.com/apache/spark/blob/master/examples/src/main/python/ml/
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, count, sum as spark_sum
from pyspark.sql.types import DoubleType, IntegerType

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

import sys

# =============================================================================
# Configuration
# =============================================================================

# HDFS Configuration
HDFS_NAMENODE = "hdfs://raspberrypi-dml0:9000"
STUDENT_ID = "810103098"  # Replace with your student ID
STUDENT_NAME = "jannati"  # Replace with your name
HDFS_INPUT_PATH = f"{HDFS_NAMENODE}/{STUDENT_NAME}/{STUDENT_ID}/Titanic-Dataset.csv"
HDFS_OUTPUT_PATH = f"{HDFS_NAMENODE}/{STUDENT_NAME}/{STUDENT_ID}/accuracy.txt"

# =============================================================================
# Initialize Spark Session
# =============================================================================

print("=" * 70)
print("DMLS - Computer Assignment 3")
print("Question 3: Titanic Data Analysis with HDFS")
print("=" * 70)

spark = SparkSession.builder \
    .appName("TitanicHDFS_Analysis") \
    .master("spark://raspberrypi-dml0:7077") \
    .config("spark.hadoop.fs.defaultFS", HDFS_NAMENODE) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print(f"\n✅ Spark Session initialized")
print(f"📌 Spark Version: {spark.version}")
print(f"📂 HDFS Input Path: {HDFS_INPUT_PATH}")

# =============================================================================
# Section 3.1: Load Data from HDFS
# =============================================================================

print("\n" + "=" * 70)
print("Section 3.1: Loading Data from HDFS")
print("=" * 70)

# Read CSV file from HDFS
df = spark.read.csv(
    HDFS_INPUT_PATH,
    header=True,
    inferSchema=True
)

# Display basic info
print(f"\n📊 Total records: {df.count()}")
print(f"📋 Columns: {df.columns}")

print("\n📝 Schema:")
df.printSchema()

print("\n📝 Sample data (first 5 rows):")
df.show(5, truncate=False)

# =============================================================================
# Section 3.2: Exploratory Data Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Section 3.2: Exploratory Data Analysis")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.2.1: Survival Rate by Sex
# -----------------------------------------------------------------------------

print("\n📊 3.2.1: Survival Rate by Sex")
print("-" * 50)

survival_by_sex = df.groupBy("Sex").agg(
    count("*").alias("Total"),
    spark_sum(col("Survived")).alias("Survived_Count"),
    (spark_sum(col("Survived")) / count("*") * 100).alias("Survival_Rate_Percent")
)

survival_by_sex.show()

# Collect results for reporting
sex_results = survival_by_sex.collect()
for row in sex_results:
    print(f"   {row['Sex']}: {row['Survival_Rate_Percent']:.2f}% survival rate "
          f"({int(row['Survived_Count'])}/{row['Total']})")

# -----------------------------------------------------------------------------
# 3.2.2: Survival Rate by Passenger Class (Pclass)
# -----------------------------------------------------------------------------

print("\n📊 3.2.2: Survival Rate by Passenger Class (Pclass)")
print("-" * 50)

survival_by_class = df.groupBy("Pclass").agg(
    count("*").alias("Total"),
    spark_sum(col("Survived")).alias("Survived_Count"),
    (spark_sum(col("Survived")) / count("*") * 100).alias("Survival_Rate_Percent")
).orderBy("Pclass")

survival_by_class.show()

# Collect results for reporting
class_results = survival_by_class.collect()
for row in class_results:
    print(f"   Class {row['Pclass']}: {row['Survival_Rate_Percent']:.2f}% survival rate "
          f"({int(row['Survived_Count'])}/{row['Total']})")

# -----------------------------------------------------------------------------
# 3.2.3: Average Age by Survival Status
# -----------------------------------------------------------------------------

print("\n📊 3.2.3: Average Age by Survival Status")
print("-" * 50)

# Filter out null ages for accurate calculation
df_with_age = df.filter(col("Age").isNotNull())

avg_age_by_survival = df_with_age.groupBy("Survived").agg(
    mean("Age").alias("Average_Age"),
    count("*").alias("Count")
).orderBy("Survived")

avg_age_by_survival.show()

# Collect results for reporting
age_results = avg_age_by_survival.collect()
for row in age_results:
    status = "Survived" if row['Survived'] == 1 else "Did Not Survive"
    print(f"   {status}: Average Age = {row['Average_Age']:.2f} years "
          f"(n={row['Count']})")

# =============================================================================
# Section 3.3: Train Logistic Regression Model
# =============================================================================

print("\n" + "=" * 70)
print("Section 3.3: Training Logistic Regression Model")
print("=" * 70)

# -----------------------------------------------------------------------------
# Step 1: Handle Missing Values
# -----------------------------------------------------------------------------

print("\n🔧 Step 1: Handling Missing Values")

# Check missing values before handling
print("\n   Missing values before handling:")
for column in ["Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"]:
    missing_count = df.filter(col(column).isNull()).count()
    print(f"   - {column}: {missing_count}")

# Handle missing values:
# - Age: Fill with median age
# - Fare: Fill with median fare
# - Embarked: Fill with mode (most common value = 'S')

# Calculate median age and fare
median_age = df.filter(col("Age").isNotNull()).approxQuantile("Age", [0.5], 0.01)[0]
median_fare = df.filter(col("Fare").isNotNull()).approxQuantile("Fare", [0.5], 0.01)[0]

print(f"\n   Median Age: {median_age:.2f}")
print(f"   Median Fare: {median_fare:.2f}")

# Apply missing value handling
df_clean = df.fillna({
    "Age": median_age,
    "Fare": median_fare,
    "Embarked": "S"  # Mode of Embarked
})

# Select only required features
df_features = df_clean.select("Pclass", "Sex", "Age", "Fare", "Embarked", "Survived")

# Drop any remaining null values
df_features = df_features.dropna()

print(f"\n   Records after cleaning: {df_features.count()}")

# -----------------------------------------------------------------------------
# Step 2: Feature Engineering
# -----------------------------------------------------------------------------

print("\n🔧 Step 2: Feature Engineering")

# Convert categorical variables to numeric using StringIndexer
# Based on: https://github.com/apache/spark/blob/master/examples/src/main/python/ml/logistic_regression_with_elastic_net.py

# Index Sex column (male=0, female=1 or vice versa)
sex_indexer = StringIndexer(
    inputCol="Sex",
    outputCol="SexIndex",
    handleInvalid="keep"
)

# Index Embarked column (S=0, C=1, Q=2 or similar)
embarked_indexer = StringIndexer(
    inputCol="Embarked",
    outputCol="EmbarkedIndex",
    handleInvalid="keep"
)

# Assemble all features into a single vector
# Features: Pclass, SexIndex, Age, Fare, EmbarkedIndex
assembler = VectorAssembler(
    inputCols=["Pclass", "SexIndex", "Age", "Fare", "EmbarkedIndex"],
    outputCol="features"
)

print("   Features used: Pclass, Sex (indexed), Age, Fare, Embarked (indexed)")

# -----------------------------------------------------------------------------
# Step 3: Split Data (80% train, 20% test)
# -----------------------------------------------------------------------------

print("\n🔧 Step 3: Splitting Data (80% Train, 20% Test)")

train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)

print(f"   Training set size: {train_data.count()}")
print(f"   Test set size: {test_data.count()}")

# -----------------------------------------------------------------------------
# Step 4: Build and Train Logistic Regression Model
# -----------------------------------------------------------------------------

print("\n🔧 Step 4: Building Logistic Regression Model")

# Create Logistic Regression model
# Based on: https://github.com/apache/spark/blob/master/examples/src/main/python/ml/logistic_regression_with_elastic_net.py

lr = LogisticRegression(
    featuresCol="features",
    labelCol="Survived",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.8
)

# Create Pipeline
pipeline = Pipeline(stages=[
    sex_indexer,
    embarked_indexer,
    assembler,
    lr
])

print("   Training model...")

# Fit the pipeline
model = pipeline.fit(train_data)

print("   ✅ Model training completed!")

# =============================================================================
# Section 3.4: Evaluate Model and Save Results
# =============================================================================

print("\n" + "=" * 70)
print("Section 3.4: Model Evaluation and Saving Results")
print("=" * 70)

# -----------------------------------------------------------------------------
# Step 1: Make Predictions
# -----------------------------------------------------------------------------

print("\n🔧 Step 1: Making Predictions on Test Data")

predictions = model.transform(test_data)

print("\n   Sample predictions:")
predictions.select("Survived", "prediction", "probability").show(10, truncate=False)

# -----------------------------------------------------------------------------
# Step 2: Calculate Accuracy
# -----------------------------------------------------------------------------

print("\n🔧 Step 2: Calculating Accuracy")

# Using MulticlassClassificationEvaluator for accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print(f"\n   ✅ Model Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Additional metrics
# Precision
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="Survived",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision = evaluator_precision.evaluate(predictions)

# Recall
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="Survived",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall = evaluator_recall.evaluate(predictions)

# F1 Score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="Survived",
    predictionCol="prediction",
    metricName="f1"
)
f1 = evaluator_f1.evaluate(predictions)

# AUC (Area Under ROC Curve)
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="Survived",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator_auc.evaluate(predictions)

print(f"   📊 Additional Metrics:")
print(f"      - Precision: {precision:.4f}")
print(f"      - Recall: {recall:.4f}")
print(f"      - F1 Score: {f1:.4f}")
print(f"      - AUC: {auc:.4f}")

# -----------------------------------------------------------------------------
# Step 3: Confusion Matrix
# -----------------------------------------------------------------------------

print("\n🔧 Step 3: Confusion Matrix")

confusion_matrix = predictions.groupBy("Survived", "prediction").count()
confusion_matrix.show()

# Calculate TP, TN, FP, FN
tp = predictions.filter((col("Survived") == 1) & (col("prediction") == 1)).count()
tn = predictions.filter((col("Survived") == 0) & (col("prediction") == 0)).count()
fp = predictions.filter((col("Survived") == 0) & (col("prediction") == 1)).count()
fn = predictions.filter((col("Survived") == 1) & (col("prediction") == 0)).count()

print(f"   True Positives (TP): {tp}")
print(f"   True Negatives (TN): {tn}")
print(f"   False Positives (FP): {fp}")
print(f"   False Negatives (FN): {fn}")

# -----------------------------------------------------------------------------
# Step 4: Save Accuracy to HDFS
# -----------------------------------------------------------------------------

print("\n🔧 Step 4: Saving Accuracy to HDFS")

# Create result text
result_text = f"""DMLS - Computer Assignment 3
Question 3: Titanic Classification Results
Student ID: {STUDENT_ID}
============================================

Model: Logistic Regression
Features: Pclass, Sex, Age, Fare, Embarked
Train/Test Split: 80/20

============================================
EVALUATION METRICS
============================================
Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}
AUC: {auc:.4f}

============================================
CONFUSION MATRIX
============================================
True Positives (TP): {tp}
True Negatives (TN): {tn}
False Positives (FP): {fp}
False Negatives (FN): {fn}

============================================
EXPLORATORY DATA ANALYSIS SUMMARY
============================================
Survival Rate by Sex:
{chr(10).join([f"  - {row['Sex']}: {row['Survival_Rate_Percent']:.2f}%" for row in sex_results])}

Survival Rate by Class:
{chr(10).join([f"  - Class {row['Pclass']}: {row['Survival_Rate_Percent']:.2f}%" for row in class_results])}

Average Age by Survival:
{chr(10).join([f"  - {'Survived' if row['Survived']==1 else 'Did Not Survive'}: {row['Average_Age']:.2f} years" for row in age_results])}
"""

# Save to HDFS using RDD
# Create RDD from result text and save as text file
result_rdd = spark.sparkContext.parallelize([result_text])
result_rdd.coalesce(1).saveAsTextFile(HDFS_OUTPUT_PATH)

print(f"\n   ✅ Results saved to: {HDFS_OUTPUT_PATH}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
📊 Analysis Complete!

1. Data loaded from HDFS: {HDFS_INPUT_PATH}
2. Exploratory Analysis:
   - Survival by Sex: Female passengers had higher survival rate
   - Survival by Class: First class had highest survival rate
   - Average Age: Survivors were slightly younger on average

3. Model Performance:
   - Accuracy: {accuracy * 100:.2f}%
   - AUC: {auc:.4f}

4. Results saved to: {HDFS_OUTPUT_PATH}
""")

# =============================================================================
# Cleanup
# =============================================================================

spark.stop()
print("\n✅ Spark Session stopped successfully!")
print("🎉 Question 3 completed!")
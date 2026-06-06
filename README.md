# Fake Review Detection Using Big Data Analytics

**Authors:** Yusuf Alperen Dönmez (1801042085) · Hüseyin Çetin (210104004245)  
**Infrastructure:** AWS EMR Serverless · Apache Spark 3.4.1 · Amazon S3  
**Dataset:** 15,544,559 Amazon product reviews (~2.5 GB compressed)

---

## Project Overview

A scalable fake review detection system that processes millions of Amazon reviews using distributed computing on AWS. The system applies a heuristic labeling strategy and compares TF-IDF + Logistic Regression vs. Random Forest for classification.

**Key Results:**

| Model | Accuracy | F1-Score | AUC-ROC | Throughput |
|-------|----------|----------|---------|------------|
| Logistic Regression | 0.9348 | 0.9117 | 0.9820 | 12,531 rec/s |
| Random Forest | 0.7403 | 0.7095 | 0.8357 | 1,713 rec/s |

---

## Repository Structure

```
fake-review-detection/
├── README.md
├── notebooks/
│   └── fake_review_pipeline.ipynb   # Main Jupyter notebook (all code + outputs)
└── paper/
    └── fake_review_detection_paper.docx
```

---

## AWS S3 Bucket Structure

Bucket name: `fake-review-detection-proje-042026` (Region: eu-north-1)

```
fake-review-detection-proje-042026/
├── raw-data/              # Original .tsv.zip files (4 datasets, ~2.5 GB)
├── unzipped-data/         # Extracted .tsv files
├── processed-data/        # Cleaned Parquet files
│   └── reviews_final_parquet/
├── models/                # Trained ML models
│   ├── lr_tfidf_model/    # Logistic Regression pipeline
│   └── rf_tfidf_model/    # Random Forest pipeline
└── notebooks/             # Jupyter notebook workspace
```

---

## AWS Infrastructure

### EMR Serverless Setup

1. **Create EMR Serverless Application**
   - AWS Console → EMR Studio → Applications
   - Type: Spark | Release: emr-6.x (Spark 3.4.1)
   - Region: eu-north-1 (Europe/Stockholm)

2. **Create EMR Studio Workspace**
   - AWS Console → EMR Studio → Workspaces → Create Workspace
   - Attach the Serverless application above

3. **Resource Configuration (Serverless — no fixed instances)**
   - Worker: 4 vCPU, 16 GB RAM per executor (auto-scaling)
   - Driver: 4 vCPU, 16 GB RAM
   - Disk: 20 GB per executor

4. **IAM Permissions**
   - EMR execution role needs `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on the bucket

---

## How to Run

### Prerequisites
- AWS account with EMR Studio access
- S3 bucket created (or use existing: `fake-review-detection-proje-042026`)
- IAM role with S3 + EMR permissions

### Step 1 — Upload Data to S3
```bash
aws s3 cp amazon_reviews_us_Automotive_v1_00.tsv.zip \
    s3://fake-review-detection-proje-042026/raw-data/
# Repeat for all 4 files
```

### Step 2 — Open Jupyter in EMR Studio
1. AWS Console → EMR Studio → Workspaces
2. Open workspace → Attach to Serverless application
3. Upload `fake_review_pipeline.ipynb`

### Step 3 — Run the Notebook

| Cell | Description |
|------|-------------|
| 1 | SparkSession init + Parquet read |
| 2 | EDA — star distribution, review length, vote stats |
| 3 | Label generation (heuristic fake/real) |
| 4 | TF-IDF + Logistic Regression (full data) |
| 5 | Random Forest (balanced sample) |
| 6 | Error analysis + interesting relationships |
| 7 | Save models to S3 |

> **Tip:** Random Forest on full data causes disk errors on default EMR config. Use the balanced 5% sample as shown in Cell 5, or increase executor disk in EMR application settings.

### Spark Configuration (recommended)
```python
spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()
```

---

## Saved Models

Trained models are saved in S3:

```
fake-review-detection-proje-042026/models/
├── lr_tfidf_model/    # Best model — AUC-ROC: 0.9820, trained on 12.4M rows (445s)
└── rf_tfidf_model/    # Comparison model — AUC-ROC: 0.8357, trained on ~798K rows (170s)
```

### Load and Use a Saved Model
```python
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Load best model
model = PipelineModel.load("s3://fake-review-detection-proje-042026/models/lr_tfidf_model")

# Predict on new reviews
new_df = spark.createDataFrame([
    ("This product is absolutely amazing!!!",),
    ("A",),
    ("Great quality, works as expected.",),
], ["review_body"])

model.transform(new_df).select("review_body", "prediction").show()
# prediction: 0 = real, 1 = fake
```

---

## Labeling Strategy

A review is labeled **fake (1)** if all three conditions hold:

1. `star_rating` in {1, 5} — extreme polarization
2. `review_len` < 20 characters — suspiciously short
3. `helpful_ratio` < 0.3 **or** null — not validated by community

Result: **7.3% fake** (1,132,301 of 15,544,559 reviews)

---

## Key Findings

- **Temporal escalation:** Fake rates near zero before 2013, jumped to 7.63% in 2014 and 12.8% in 2015
- **Asymmetric manipulation:** 5-star fake rate (10.93%) is 2x higher than 1-star (4.93%)
- **High-frequency users:** Top reviewer submitted 2,780 reviews — likely automated
- **LR vs RF:** Linear models outperform tree-based on sparse TF-IDF features (7.3x faster)

---

## Dataset

Amazon Customer Reviews Dataset (public):

| Category | Download Link |
|----------|--------------|
| Automotive | https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Automotive_v1_00.tsv.gz |
| Digital Video Download | https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz |
| Health & Personal Care | https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Health_Personal_Care_v1_00.tsv.gz |
| Pet Products | https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Pet_Products_v1_00.tsv.gz |

> Veriyi indirmeden doğrudan Spark ile okumak için:
```python
df = spark.read.option("delimiter", "\t") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("s3://amazon-reviews-pds/tsv/amazon_reviews_us_Automotive_v1_00.tsv.gz")
```

---

## Dependencies

All dependencies are pre-installed in EMR Serverless Spark 3.4.1:
- PySpark 3.4.1
- boto3
- zipfile (stdlib)

No additional installations required.

After the models were trained, they were saved under the s3://fake-review-detection-proje-042026/models/ directory in order to preserve the distributed architecture.

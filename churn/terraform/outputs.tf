output "s3_raw_data_bucket" {
  value       = aws_s3_bucket.data_lake.bucket
  description = "S3 bucket name for raw data storage"
}

output "s3_churn_results_bucket" {
  value       = aws_s3_bucket.churn_results.bucket
  description = "S3 bucket for churn prediction results and dbt outputs"
}

output "s3_ml_models_bucket" {
  value       = aws_s3_bucket.ml_models.bucket
  description = "S3 bucket for ML model artifacts and trained models"
}

output "s3_bucket_urls" {
  value = {
    raw_data = "s3://${aws_s3_bucket.data_lake.bucket}"
    churn_results = "s3://${aws_s3_bucket.churn_results.bucket}"
    ml_models = "s3://${aws_s3_bucket.ml_models.bucket}"
  }
  description = "S3 bucket URLs for all data storage buckets"
}
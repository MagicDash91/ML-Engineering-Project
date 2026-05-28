resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-raw-data"

  tags = {
    Name    = "${var.project_name}-raw-data"
    Project = var.project_name
  }
}

resource "aws_s3_bucket_versioning" "data_lake_versioning" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket for churn prediction results
resource "aws_s3_bucket" "churn_results" {
  bucket = "${var.project_name}-churn-results"

  tags = {
    Name    = "${var.project_name}-churn-results"
    Project = var.project_name
    Purpose = "churn-prediction-outputs"
  }
}

resource "aws_s3_bucket_versioning" "churn_results_versioning" {
  bucket = aws_s3_bucket.churn_results.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "churn_results_encryption" {
  bucket = aws_s3_bucket.churn_results.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 bucket for model artifacts
resource "aws_s3_bucket" "ml_models" {
  bucket = "${var.project_name}-ml-models"

  tags = {
    Name    = "${var.project_name}-ml-models"
    Project = var.project_name
    Purpose = "ml-model-artifacts"
  }
}

resource "aws_s3_bucket_versioning" "ml_models_versioning" {
  bucket = aws_s3_bucket.ml_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_models_encryption" {
  bucket = aws_s3_bucket.ml_models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
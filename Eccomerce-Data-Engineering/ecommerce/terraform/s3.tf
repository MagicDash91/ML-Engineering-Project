# -------------------------------------------------------------------------- #
# S3 landing bucket for raw Olist CSV snapshots (lineage / DR copy).
# force_destroy = true so `terraform destroy` succeeds even when non-empty.
# -------------------------------------------------------------------------- #
resource "aws_s3_bucket" "raw_data" {
  bucket        = "${var.project_name}-raw-data"
  force_destroy = true

  tags = {
    Name    = "${var.project_name}-raw-data"
    Purpose = "olist-raw-landing"
  }
}

resource "aws_s3_bucket_versioning" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "raw_data" {
  bucket                  = aws_s3_bucket.raw_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Expire raw snapshots after 30 days to keep storage cost negligible.
resource "aws_s3_bucket_lifecycle_configuration" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id
  rule {
    id     = "expire-raw-snapshots"
    status = "Enabled"
    filter {
      prefix = "olist/raw/"
    }
    expiration {
      days = 30
    }
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

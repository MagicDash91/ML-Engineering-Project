# -------------------------------------------------------------------------- #
# Least-privilege IAM role for the Airflow EC2 host: read/write only to the
# project's own S3 landing bucket.
# -------------------------------------------------------------------------- #
data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "airflow" {
  name               = "${var.project_name}-airflow-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

data "aws_iam_policy_document" "s3_access" {
  statement {
    sid       = "ListLandingBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.raw_data.arn]
  }
  statement {
    sid       = "ReadWriteLandingObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.raw_data.arn}/*"]
  }
}

resource "aws_iam_role_policy" "s3_access" {
  name   = "${var.project_name}-s3-access"
  role   = aws_iam_role.airflow.id
  policy = data.aws_iam_policy_document.s3_access.json
}

resource "aws_iam_instance_profile" "airflow" {
  name = "${var.project_name}-airflow-profile"
  role = aws_iam_role.airflow.name
}

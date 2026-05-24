variable "aws_region" {
  description = "AWS region for S3 buckets"
  default     = "ap-southeast-1"
}

variable "project_name" {
  description = "Project name used as prefix for S3 bucket names"
  default     = "magicdash-data-pipeline"
}
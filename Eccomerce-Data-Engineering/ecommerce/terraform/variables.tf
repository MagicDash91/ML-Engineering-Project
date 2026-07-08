variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "ap-southeast-1"
}

variable "project_name" {
  description = "Prefix used for resource names (must be globally unique for S3)."
  type        = string
  default     = "magicdash-olist-ecommerce"
}

variable "instance_type" {
  description = "EC2 instance type for the self-hosted Airflow host. t3.medium (4 GB) is the smallest size that reliably boots the full Astro/Airflow stack."
  type        = string
  default     = "t3.medium"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access (leave empty to skip)."
  type        = string
  default     = ""
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to SSH / reach the Airflow UI. Lock this to your IP."
  type        = string
  default     = "0.0.0.0/0"
}

variable "budget_alert_email" {
  description = "Email address that receives AWS Budgets alerts."
  type        = string
  default     = "michwirja@gmail.com"
}

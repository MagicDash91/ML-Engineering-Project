output "raw_data_bucket" {
  description = "S3 landing bucket for raw Olist snapshots."
  value       = aws_s3_bucket.raw_data.bucket
}

output "airflow_instance_id" {
  description = "EC2 instance id of the Airflow host (stop it between runs)."
  value       = aws_instance.airflow.id
}

output "airflow_public_ip" {
  description = "Public IP of the Airflow host (Airflow UI on :8080)."
  value       = aws_instance.airflow.public_ip
}

output "airflow_iam_role" {
  description = "IAM role assumed by the Airflow host."
  value       = aws_iam_role.airflow.name
}

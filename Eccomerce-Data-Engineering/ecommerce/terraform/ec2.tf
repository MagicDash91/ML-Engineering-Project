# -------------------------------------------------------------------------- #
# Self-hosted Airflow host. Stop the instance between DAG runs to control cost
# (see README teardown / cost notes).
# -------------------------------------------------------------------------- #
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_security_group" "airflow" {
  name        = "${var.project_name}-airflow-sg"
  description = "Airflow host: SSH + Airflow UI from the allowed CIDR only."

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  ingress {
    description = "Airflow UI"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-airflow-sg"
  }
}

resource "aws_instance" "airflow" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.airflow.name
  vpc_security_group_ids = [aws_security_group.airflow.id]
  key_name               = var.key_pair_name != "" ? var.key_pair_name : null

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    dnf update -y
    dnf install -y docker git python3-pip
    systemctl enable --now docker
    usermod -aG docker ec2-user
    # Install the Astro CLI so `astro dev start` can run this project on the host.
    curl -sSL https://install.astronomer.io | bash -s
  EOF

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.project_name}-airflow"
  }
}

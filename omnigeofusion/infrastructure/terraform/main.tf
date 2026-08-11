# ================================================================
# OmniGeoFusion — Terraform Infrastructure
# ================================================================
# Free-tier compatible infrastructure for development
#
# Resources created:
#   Networking: VPC, subnets, IGW, NAT, route tables, S3 endpoint
#   Storage:    S3 bucket + lifecycle + ECR repositories
#   Compute:    EC2 t2.micro (free tier) + Elastic IP
#   Database:   RDS PostgreSQL 17.10 db.t3.micro
#   Cache:      ElastiCache Redis cache.t3.micro
#   Monitoring: CloudWatch alarms + AWS Budget alert
#
# EKS intentionally excluded:
#   EKS worker nodes require t3.medium minimum
#   which is not available on free tier accounts
#   API will be served directly from EC2 instead
#   EKS will be added in P3 with paid account
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply
#   terraform destroy  (between sessions to save cost)
# ================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}


# ── Data sources ──────────────────────────────────────────────
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}


# ── Locals ────────────────────────────────────────────────────
locals {
  name       = "${var.project_name}-${var.environment}"
  account_id = data.aws_caller_identity.current.account_id
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}


# ================================================================
# NETWORKING
# ================================================================

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = merge(local.tags, { Name = "${local.name}-vpc" })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "${local.name}-igw" })
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = merge(local.tags, {
    Name = "${local.name}-public-${count.index + 1}"
  })
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = merge(local.tags, {
    Name = "${local.name}-private-${count.index + 1}"
  })
}

resource "aws_eip" "nat" {
  domain     = "vpc"
  depends_on = [aws_internet_gateway.main]
  tags       = merge(local.tags, { Name = "${local.name}-nat-eip" })
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.main]
  tags          = merge(local.tags, { Name = "${local.name}-nat" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = merge(local.tags, { Name = "${local.name}-public-rt" })
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  tags = merge(local.tags, { Name = "${local.name}-private-rt" })
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [
    aws_route_table.public.id,
    aws_route_table.private.id
  ]
  tags = merge(local.tags, { Name = "${local.name}-s3-endpoint" })
}


# ================================================================
# SECURITY GROUPS
# ================================================================

resource "aws_security_group" "ec2" {
  name        = "${local.name}-ec2-sg"
  description = "OmniGeoFusion EC2 security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "Airflow Web UI"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "MLflow"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "FastAPI"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(local.tags, { Name = "${local.name}-ec2-sg" })
}

resource "aws_security_group" "rds" {
  name        = "${local.name}-rds-sg"
  description = "RDS PostgreSQL security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from EC2"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(local.tags, { Name = "${local.name}-rds-sg" })
}

resource "aws_security_group" "redis" {
  name        = "${local.name}-redis-sg"
  description = "ElastiCache Redis security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Redis from EC2"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(local.tags, { Name = "${local.name}-redis-sg" })
}


# ================================================================
# STORAGE
# ================================================================

resource "aws_s3_bucket" "data" {
  bucket        = "${var.project_name}-data-${local.account_id}"
  force_destroy = true
  tags = merge(local.tags, {
    Name      = "${local.name}-data"
    Purpose   = "Netherlands multimodal geospatial data"
    Persistent = "true"
  })
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "delete-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

resource "aws_s3_object" "folders" {
  for_each = toset([
    "netherlands/sentinel2/",
    "netherlands/sentinel1/",
    "netherlands/lidar/",
    "netherlands/thermal/",
    "netherlands/iot/",
    "netherlands/osm/",
    "netherlands/pairs/",
    "netherlands/targets/",
    "netherlands/stats/",
    "models/ssl/",
    "models/finetune/",
    "models/best/",
    "repo/",
    "logs/",
  ])
  bucket  = aws_s3_bucket.data.id
  key     = each.value
  content = ""
}

resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}/api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = merge(local.tags, {
    Name      = "${local.name}-api-ecr"
    Persistent = "true"
  })
}

resource "aws_ecr_repository" "fusion" {
  name                 = "${var.project_name}/fusion"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = merge(local.tags, {
    Name      = "${local.name}-fusion-ecr"
    Persistent = "true"
  })
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_ecr_lifecycle_policy" "fusion" {
  repository = aws_ecr_repository.fusion.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}


# ================================================================
# DATABASE
# ================================================================

resource "aws_db_subnet_group" "main" {
  name       = "${local.name}-db-subnet"
  subnet_ids = aws_subnet.private[*].id
  tags       = merge(local.tags, { Name = "${local.name}-db-subnet" })
}

resource "aws_db_instance" "postgres" {
  identifier            = "${local.name}-postgres"
  engine                = "postgres"
  engine_version        = "17.10"
  instance_class        = "db.t3.micro"
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  db_name  = "omnigeofusion"
  username = "omnigeofusion_admin"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false
  multi_az               = false

  backup_retention_period  = 1
  skip_final_snapshot      = true
  delete_automated_backups = true
  deletion_protection      = false

  backup_window      = "03:00-04:00"
  maintenance_window = "Mon:04:00-Mon:05:00"

  tags = merge(local.tags, {
    Name    = "${local.name}-postgres"
    Purpose = "PostGIS patch metadata OSM vectors predictions"
  })
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.name}-redis-subnet"
  subnet_ids = aws_subnet.private[*].id
  tags       = merge(local.tags, { Name = "${local.name}-redis-subnet" })
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${local.name}-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]

  tags = merge(local.tags, {
    Name    = "${local.name}-redis"
    Purpose = "Fusion embedding cache and session store"
  })
}


# ================================================================
# COMPUTE — EC2 (t2.micro free tier)
# ================================================================

resource "aws_iam_role" "ec2" {
  name = "${local.name}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
  tags = merge(local.tags, { Name = "${local.name}-ec2-role" })
}

resource "aws_iam_role_policy_attachment" "ec2_s3" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "ec2_ecr" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
}

resource "aws_iam_role_policy_attachment" "ec2_cloudwatch" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${local.name}-ec2-profile"
  role = aws_iam_role.ec2.name
}

resource "aws_key_pair" "main" {
  key_name   = "${local.name}-key"
  public_key = var.ec2_public_key
  tags       = merge(local.tags, { Name = "${local.name}-key" })
}

resource "aws_instance" "main" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = "t3.micro"
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.ec2.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name
  key_name               = aws_key_pair.main.key_name

  root_block_device {
    volume_size           = 30
    volume_type           = "gp2"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = base64encode(<<-USERDATA
    #!/bin/bash
    set -e
    echo "=== OmniGeoFusion EC2 Setup ==="

    apt-get update -y
    apt-get install -y \
      python3-pip python3-venv python3-dev \
      docker.io docker-compose \
      awscli git curl wget unzip \
      htop ncdu tree jq \
      build-essential gcc g++ \
      gdal-bin libgdal-dev \
      libgeos-dev libproj-dev \
      libspatialindex-dev \
      libpq-dev postgresql-client

    systemctl enable docker
    systemctl start docker
    usermod -aG docker ubuntu

    aws configure set region ${var.aws_region}

    sudo -u ubuntu python3 -m venv /home/ubuntu/venv
    sudo -u ubuntu /home/ubuntu/venv/bin/pip install \
      --upgrade pip wheel

    sudo -u ubuntu /home/ubuntu/venv/bin/pip install \
      boto3 rasterio geopandas shapely \
      numpy pandas scipy scikit-learn \
      mlflow psycopg2-binary \
      apache-airflow==2.8.0 \
      apache-airflow-providers-amazon \
      tqdm pyyaml requests

    mkdir -p /home/ubuntu/mlflow
    chown ubuntu:ubuntu /home/ubuntu/mlflow

    cat > /home/ubuntu/start_services.sh << 'STARTUP'
#!/bin/bash
source /home/ubuntu/venv/bin/activate

nohup mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///home/ubuntu/mlflow/mlflow.db \
  --default-artifact-root s3://omnigeofusion-data-${local.account_id}/models \
  > /home/ubuntu/mlflow/server.log 2>&1 &

echo "MLflow started on port 5000"

export AIRFLOW_HOME=/home/ubuntu/airflow
nohup airflow webserver --port 8080 \
  > /home/ubuntu/airflow/webserver.log 2>&1 &
nohup airflow scheduler \
  > /home/ubuntu/airflow/scheduler.log 2>&1 &

echo "Airflow started on port 8080"
STARTUP

    chmod +x /home/ubuntu/start_services.sh
    chown ubuntu:ubuntu /home/ubuntu/start_services.sh

    echo "=== Setup complete ==="
  USERDATA
  )

  tags = merge(local.tags, {
    Name    = "${local.name}-ec2"
    Purpose = "Data processing Airflow MLflow"
  })
}

resource "aws_eip" "main" {
  instance   = aws_instance.main.id
  domain     = "vpc"
  depends_on = [aws_internet_gateway.main]
  tags       = merge(local.tags, { Name = "${local.name}-eip" })
}


# ================================================================
# CLOUDWATCH MONITORING
# ================================================================

resource "aws_cloudwatch_metric_alarm" "ec2_cpu" {
  alarm_name          = "${local.name}-ec2-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "EC2 CPU above 85 percent for 10 minutes"
  dimensions          = { InstanceId = aws_instance.main.id }
  tags                = local.tags
}

resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  alarm_name          = "${local.name}-rds-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "RDS CPU above 80 percent"
  dimensions          = { DBInstanceIdentifier = aws_db_instance.postgres.id }
  tags                = local.tags
}

resource "aws_cloudwatch_metric_alarm" "rds_storage" {
  alarm_name          = "${local.name}-rds-storage-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 2000000000
  alarm_description   = "RDS free storage below 2GB"
  dimensions          = { DBInstanceIdentifier = aws_db_instance.postgres.id }
  tags                = local.tags
}


# ================================================================
# AWS BUDGET ALARM
# ================================================================

resource "aws_budgets_budget" "main" {
  name         = "${local.name}-budget"
  budget_type  = "COST"
  limit_amount = "30"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}

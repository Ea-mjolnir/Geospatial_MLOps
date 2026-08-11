# ================================================================
# OmniGeoFusion — Terraform Outputs
# ================================================================

# ── Networking ────────────────────────────────────────────────
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

# ── S3 ────────────────────────────────────────────────────────
output "s3_bucket_name" {
  description = "S3 data bucket name"
  value       = aws_s3_bucket.data.id
}

output "s3_bucket_arn" {
  description = "S3 data bucket ARN"
  value       = aws_s3_bucket.data.arn
}

# ── ECR ───────────────────────────────────────────────────────
output "ecr_api_url" {
  description = "ECR repository URL for API image"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_fusion_url" {
  description = "ECR repository URL for fusion image"
  value       = aws_ecr_repository.fusion.repository_url
}

# ── RDS ───────────────────────────────────────────────────────
output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS PostgreSQL port"
  value       = aws_db_instance.postgres.port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = aws_db_instance.postgres.db_name
}

output "postgres_connection_string" {
  description = "PostgreSQL connection string"
  value       = "postgresql://omnigeofusion_admin:PASSWORD@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
  sensitive   = true
}

# ── Redis ─────────────────────────────────────────────────────
output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_cluster.redis.cache_nodes[0].address
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_cluster.redis.port
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:${aws_elasticache_cluster.redis.port}"
}

# ── EC2 ───────────────────────────────────────────────────────
output "ec2_public_ip" {
  description = "EC2 instance public IP"
  value       = aws_eip.main.public_ip
}

output "ec2_instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.main.id
}

output "ec2_ssh_command" {
  description = "SSH command for EC2 instance"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_eip.main.public_ip}"
}

output "mlflow_url" {
  description = "MLflow tracking server URL"
  value       = "http://${aws_eip.main.public_ip}:5000"
}

output "airflow_url" {
  description = "Airflow web UI URL"
  value       = "http://${aws_eip.main.public_ip}:8080"
}

# ── Summary ───────────────────────────────────────────────────
output "deployment_summary" {
  description = "OmniGeoFusion deployment summary"
  value = {
    project     = var.project_name
    environment = var.environment
    region      = var.aws_region
    s3_bucket   = aws_s3_bucket.data.id
    ec2_ip      = aws_eip.main.public_ip
    mlflow      = "http://${aws_eip.main.public_ip}:5000"
    airflow     = "http://${aws_eip.main.public_ip}:8080"
    ecr_api     = aws_ecr_repository.api.repository_url
    rds         = aws_db_instance.postgres.endpoint
    redis       = aws_elasticache_cluster.redis.cache_nodes[0].address
  }
  sensitive = true
}

# ================================================================
# OmniGeoFusion — Terraform Variables
# ================================================================

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"

  validation {
    condition = contains([
      "us-east-1", "us-west-2", "eu-west-1", "eu-central-1"
    ], var.aws_region)
    error_message = "Region must be us-east-1, us-west-2, eu-west-1 or eu-central-1."
  }
}

variable "project_name" {
  description = "Project name used as prefix for all resources"
  type        = string
  default     = "omnigeofusion"

  validation {
    condition     = length(var.project_name) <= 20
    error_message = "Project name must be 20 chars or less."
  }
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging or prod."
  }
}

variable "db_password" {
  description = "RDS PostgreSQL master password (min 8 chars, alphanumeric only)"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.db_password) >= 8 && can(regex("^[a-zA-Z0-9]+$", var.db_password))
    error_message = "DB password must be at least 8 alphanumeric characters. No special characters."
  }
}

variable "ec2_public_key" {
  description = "SSH public key for EC2 access"
  type        = string
}

variable "alert_email" {
  description = "Email address for AWS Budget and CloudWatch alerts"
  type        = string

  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.alert_email))
    error_message = "Must be a valid email address."
  }
}

variable "eks_node_type" {
  description = "EKS worker node instance type (used when EKS is enabled)"
  type        = string
  default     = "t3.medium"
}

variable "eks_min_nodes" {
  description = "Minimum EKS worker nodes (used when EKS is enabled)"
  type        = number
  default     = 1
}

variable "eks_max_nodes" {
  description = "Maximum EKS worker nodes (used when EKS is enabled)"
  type        = number
  default     = 3
}

variable "ec2_instance_type" {
  description = "EC2 instance type for data processing"
  type        = string
  default     = "t2.micro"
}

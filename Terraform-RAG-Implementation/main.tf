# main.tf
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "project_name" {
  default = "myproject"
}

variable "location" {
  default = "southeastasia"
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-${var.project_name}"
  location = var.location
  tags = {
    environment = "dev"
    managed_by  = "terraform"
  }
}

# Azure Container Registry
resource "azurerm_container_registry" "main" {
  name                = "acrragollama"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
}

# Container App Environment
resource "azurerm_container_app_environment" "main" {
  name                = "cae-${var.project_name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
}

# Container App
resource "azurerm_container_app" "main" {
  name                         = "ca-${var.project_name}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name   = "rag-ollama"
      image  = "${azurerm_container_registry.main.login_server}/rag-ollama:latest"
      # Ollama + model butuh resource cukup besar
      cpu    = 2.0
      memory = "4Gi"
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  registry {
    server               = azurerm_container_registry.main.login_server
    username             = azurerm_container_registry.main.admin_username
    password_secret_name = "acr-password"
  }

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.main.admin_password
  }
}

# Outputs
output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "acr_login_server" {
  value = azurerm_container_registry.main.login_server
}

output "app_url" {
  value = "https://${azurerm_container_app.main.latest_revision_fqdn}"
}
# AWS Cloud Infrastructure with CAG System

A production-ready AWS infrastructure setup featuring Virtual Private Cloud (VPC) configuration, public/private subnets, NAT Gateway, and a deployed FastAPI-based Context Augmented Generation (CAG) system powered by Gemini AI.

## 🏗️ Architecture Overview

This project demonstrates a complete AWS cloud infrastructure implementation with security best practices and scalable architecture patterns.

### Infrastructure Components

**Network Layer**
- Custom VPC with CIDR block `10.0.0.0/16`
- Public subnet (`10.0.1.0/24`) for internet-facing resources
- Private subnet (`10.0.2.0/24`) for internal resources
- Internet Gateway for public internet access
- NAT Gateway for private subnet outbound connectivity
- Custom route tables with proper associations

**Compute & Security**
- EC2 instance (t2.micro) running Ubuntu 24.04 LTS
- Security group with SSH access configured
- Public IPv4 address: `47.129.154.23`
- Private IPv4 address: `10.0.1.242`

**Application**
- FastAPI web application for Context Augmented Generation
- Integration with Google's Gemini AI
- RESTful API endpoints for context management and AI queries

## 📋 Features

### AWS Infrastructure

- **Isolated Network Environment**: Complete VPC setup with public and private subnets
- **Internet Connectivity**: Internet Gateway attached to VPC for public access
- **NAT Gateway**: Enables private subnet resources to access the internet securely
- **Route Table Management**: Properly configured routing for both public and private traffic
- **Security Groups**: Firewall rules controlling inbound and outbound traffic
- **Resource Tagging**: Organized resource management with consistent naming conventions

### CAG System Application

- **Context Management**: Add and store contextual information with unique IDs
- **AI-Powered Queries**: Ask questions based on stored context using Gemini AI
- **RESTful API**: Clean API design with FastAPI framework
- **Persistent Storage**: Context information stored and retrieved efficiently

## 🚀 Deployment Architecture

```
Internet
    ↓
Internet Gateway (igw-cloud)
    ↓
VPC (vpc-cloud) - 10.0.0.0/16
    ├── Public Subnet (subnet-public-cloud) - 10.0.1.0/24
    │   ├── EC2 Instance (server1)
    │   │   └── FastAPI Application (Port 8000)
    │   └── NAT Gateway
    │
    └── Private Subnet (subnet-private-cloud) - 10.0.2.0/24
        └── Future backend resources

Route Tables:
- Public: Routes to Internet Gateway
- Main: Associated with private subnet
```

## 💻 Technical Stack

**Infrastructure**
- AWS VPC
- AWS EC2
- AWS Internet Gateway
- AWS NAT Gateway
- AWS Route Tables

**Application**
- Python 3.x
- FastAPI framework
- Uvicorn ASGI server
- Google Gemini AI API
- python-dotenv for environment management

## 📦 Installation & Setup

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured
- Python 3.8+
- Google Gemini API key

### Infrastructure Setup

**1. VPC Creation**
```bash
# VPC with CIDR 10.0.0.0/16
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=vpc-cloud}]'
```

**2. Subnet Creation**
```bash
# Public subnet
aws ec2 create-subnet --vpc-id <vpc-id> --cidr-block 10.0.1.0/24 --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=subnet-public-cloud}]'

# Private subnet
aws ec2 create-subnet --vpc-id <vpc-id> --cidr-block 10.0.2.0/24 --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=subnet-private-cloud}]'
```

**3. Internet Gateway**
```bash
# Create and attach IGW
aws ec2 create-internet-gateway --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=igw-cloud}]'
aws ec2 attach-internet-gateway --vpc-id <vpc-id> --internet-gateway-id <igw-id>
```

**4. NAT Gateway**
```bash
# Allocate Elastic IP and create NAT Gateway
aws ec2 allocate-address --domain vpc
aws ec2 create-nat-gateway --subnet-id <public-subnet-id> --allocation-id <eip-allocation-id>
```

**5. Route Tables**
```bash
# Create route table and add routes
aws ec2 create-route-table --vpc-id <vpc-id> --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=route-tables}]'
aws ec2 create-route --route-table-id <rtb-id> --destination-cidr-block 0.0.0.0/0 --gateway-id <igw-id>
```

### Application Deployment

**1. Launch EC2 Instance**
```bash
aws ec2 run-instances \
  --image-id ami-00d8fc944b171e8c2 \
  --instance-type t2.micro \
  --subnet-id <subnet-id> \
  --security-group-ids <sg-id> \
  --key-name <your-key-pair> \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=server1}]'
```

**2. Connect to Instance**
```bash
ssh -i your-key.pem ubuntu@47.129.154.23
```

**3. Install Dependencies**
```bash
# Update system
sudo apt update

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install fastapi uvicorn python-dotenv google-generativeai
```

**4. Setup Application**
```bash
# Create application directory
mkdir fastapi && cd fastapi

# Create .env file with your Gemini API key
nano .env
# Add: GEMINI_API_KEY=your_api_key_here

# Create main.py with your FastAPI application code
nano main.py
```

**5. Run Application**
```bash
# Start the FastAPI server
python main.py
```

## 🔧 Application API Endpoints

### Add Context
**POST** `/add-context`

Add contextual information for AI to reference.

**Request Body:**
```json
{
  "context_id": "product-info",
  "context_info": "Our company sells eco-friendly water bottles..."
}
```

**Response:**
```json
{
  "message": "Context added successfully",
  "context_id": "product-info"
}
```

### Ask Question
**POST** `/ask`

Query the AI using stored context.

**Request Body:**
```json
{
  "context_id": "product-info",
  "question": "What products does the company sell?"
}
```

**Response:**
```json
{
  "context_id": "product-info",
  "question": "What products does the company sell?",
  "answer": "The company sells eco-friendly water bottles..."
}
```

### List Contexts
**GET** `/contexts`

Retrieve all stored context IDs.

**Response:**
```json
{
  "contexts": ["product-info", "company-policy", "faq"]
}
```

## 🔒 Security Considerations

**Network Security**
- Private subnet for sensitive resources
- NAT Gateway for secure outbound connections from private subnet
- Security groups restricting access to necessary ports only
- SSH access limited to port 22

**Application Security**
- Environment variables for API keys
- No hardcoded credentials
- HTTPS recommended for production (configure with SSL/TLS)

**Best Practices Implemented**
- Principle of least privilege for IAM roles
- Resource tagging for cost tracking and management
- Proper subnet isolation (public/private)
- Centralized route table management

## 📊 Resource Configuration

| Resource | Configuration | Purpose |
|----------|--------------|---------|
| VPC | 10.0.0.0/16 | Isolated network environment |
| Public Subnet | 10.0.1.0/24 | Internet-accessible resources |
| Private Subnet | 10.0.2.0/24 | Internal resources |
| EC2 Instance | t2.micro, Ubuntu 24.04 | Application server |
| Internet Gateway | Attached to VPC | Public internet access |
| NAT Gateway | In public subnet | Private subnet internet access |

## 🌐 Access Information

- **Application URL**: `http://47.129.154.23:8000`
- **Region**: ap-southeast-1 (Singapore)
- **Availability Zone**: ap-southeast-1b
- **Instance Type**: t2.micro
- **Operating System**: Ubuntu Server 24.04 LTS

## 📈 Future Enhancements

- [ ] Implement Application Load Balancer for high availability
- [ ] Add Auto Scaling Groups for automatic scaling
- [ ] Deploy database in private subnet (RDS)
- [ ] Implement CloudWatch monitoring and alarms
- [ ] Add S3 bucket for static assets
- [ ] Configure Route 53 for custom domain
- [ ] Implement CI/CD pipeline with AWS CodePipeline
- [ ] Add CloudFront CDN for content delivery
- [ ] Implement VPC Flow Logs for network monitoring
- [ ] Setup AWS WAF for application protection

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- AWS Documentation for infrastructure best practices
- FastAPI framework for excellent API development experience
- Google Gemini AI for powerful language model capabilities
- Ubuntu community for stable server operating system

---

⭐ If you find this project useful, please consider giving it a star!

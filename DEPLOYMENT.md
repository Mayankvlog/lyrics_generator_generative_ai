# Deployment Guide

Complete guide for deploying the Lyrics Generator to production environments.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Prerequisites](#prerequisites)
- [Local/Development Deployment](#localdevelopment-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platforms](#cloud-platforms)
- [Security Considerations](#security-considerations)
- [Monitoring and Logging](#monitoring-and-logging)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## Deployment Options

| Method | Complexity | Scalability | Cost | Best For |
|--------|-----------|------------|------|----------|
| Local/Development | Low | None | Free | Development & Testing |
| Docker | Low | Limited | Free/Low | Small-Medium Projects |
| Kubernetes | High | Excellent | Medium | Large-Scale Projects |
| AWS/GCP/Azure | Medium-High | Excellent | Variable | Enterprise Solutions |
| Heroku | Low | Limited | Medium | Quick Deployment |

## Prerequisites

### All Methods
- Git installed
- Repository cloned
- Environment variables configured
- Model files available

### Docker Methods
- Docker 20.10+
- Docker Compose 2.0+ (optional)

### Kubernetes
- Kubernetes cluster (1.20+)
- kubectl configured
- Helm (optional)

### Cloud Platforms
- Cloud account and CLI configured
- Container registry access
- Domain name (optional but recommended)

## Local/Development Deployment

### Simple HTTP Server

```bash
# 1. Install and activate
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Set environment
export MONGODB_URI="your_connection_string"

# 3. Run application
python -m streamlit run main.py --server.port 8501 --server.address 0.0.0.0
```

### systemd Service (Linux)

Create `/etc/systemd/system/lyrics-generator.service`:

```ini
[Unit]
Description=Lyrics Generator Streamlit App
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/lyrics_generator
Environment="MONGODB_URI=your_connection_string"
ExecStart=/opt/lyrics_generator/myenv/bin/python -m streamlit run main.py --server.port 8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable lyrics-generator
sudo systemctl start lyrics-generator
```

## Docker Deployment

### Single Container Deployment

```bash
# 1. Build image
docker build -t lyrics-generator:latest .

# 2. Create .env file
cat > .env << EOF
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/food
EOF

# 3. Run container
docker run -d \
  --name lyrics-app \
  -p 8502:8502 \
  --env-file .env \
  --restart unless-stopped \
  lyrics-generator:latest

# 4. View logs
docker logs -f lyrics-app

# 5. Stop container
docker stop lyrics-app
```

### Docker Compose Production Setup

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    image: lyrics-generator:latest
    container_name: lyrics-generator-app
    restart: unless-stopped
    ports:
      - "8502:8502"
    environment:
      - MONGODB_URI=${MONGODB_URI}
      - STREAMLIT_SERVER_PORT=8502
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_BROWSER_GATHERUSAGESTATS=false
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8502"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - lyrics-network
    depends_on:
      - mongodb

  mongodb:
    image: mongo:latest
    container_name: lyrics-mongodb
    restart: unless-stopped
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGO_ROOT_USER}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_ROOT_PASSWORD}
    volumes:
      - mongodb_data:/data/db
    networks:
      - lyrics-network
    healthcheck:
      test: ["CMD", "mongo", "--eval", "db.adminCommand('ping')"]
      interval: 10s
      timeout: 5s
      retries: 5

  nginx:
    image: nginx:alpine
    container_name: lyrics-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    networks:
      - lyrics-network
    depends_on:
      - app

volumes:
  mongodb_data:

networks:
  lyrics-network:
    driver: bridge
```

Deploy with:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster running
- kubectl configured
- Container registry (Docker Hub, ECR, GCR, etc.)

### Step 1: Prepare Docker Image

```bash
# Build and push to registry
docker build -t your-registry/lyrics-generator:latest .
docker push your-registry/lyrics-generator:latest
```

### Step 2: Create Kubernetes Manifests

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lyrics-generator
  labels:
    app: lyrics-generator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lyrics-generator
  template:
    metadata:
      labels:
        app: lyrics-generator
    spec:
      containers:
      - name: app
        image: your-registry/lyrics-generator:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8502
        env:
        - name: MONGODB_URI
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: mongodb-uri
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /
            port: 8502
          initialDelaySeconds: 40
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8502
          initialDelaySeconds: 20
          periodSeconds: 5
```

Create `k8s/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: lyrics-generator-service
spec:
  type: LoadBalancer
  selector:
    app: lyrics-generator
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8502
```

Create `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  STREAMLIT_SERVER_HEADLESS: "true"
  STREAMLIT_BROWSER_GATHERUSAGESTATS: "false"
```

### Step 3: Deploy

```bash
# Create namespace
kubectl create namespace lyrics-generator

# Create secrets
kubectl create secret generic app-secrets \
  --from-literal=mongodb-uri="your_mongodb_uri" \
  -n lyrics-generator

# Apply manifests
kubectl apply -f k8s/configmap.yaml -n lyrics-generator
kubectl apply -f k8s/deployment.yaml -n lyrics-generator
kubectl apply -f k8s/service.yaml -n lyrics-generator

# Check deployment
kubectl get deployments -n lyrics-generator
kubectl get pods -n lyrics-generator
kubectl get services -n lyrics-generator
```

### Step 4: Access Application

```bash
# Get external IP
kubectl get service lyrics-generator-service -n lyrics-generator

# Port forward (local testing)
kubectl port-forward svc/lyrics-generator-service 8502:80 -n lyrics-generator
```

## Cloud Platforms

### AWS Elastic Container Service (ECS)

```bash
# 1. Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com
docker tag lyrics-generator:latest [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/lyrics-generator:latest
docker push [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/lyrics-generator:latest

# 2. Create ECS task definition
# Use AWS Console or CLI with task definition JSON

# 3. Create ECS service
aws ecs create-service --cluster lyrics-cluster --service-name lyrics-service --task-definition lyrics-task --desired-count 3
```

### Google Cloud Run

```bash
# 1. Build and push
gcloud builds submit --tag gcr.io/PROJECT-ID/lyrics-generator

# 2. Deploy
gcloud run deploy lyrics-generator \
  --image gcr.io/PROJECT-ID/lyrics-generator \
  --platform managed \
  --region us-central1 \
  --set-env-vars MONGODB_URI="your_uri"

# 3. Access
gcloud run services describe lyrics-generator --platform managed --region us-central1
```

### Heroku

```bash
# 1. Create Procfile
echo "web: streamlit run main.py --server.port \$PORT --server.address 0.0.0.0" > Procfile

# 2. Login and create app
heroku login
heroku create lyrics-generator-app

# 3. Set environment variables
heroku config:set MONGODB_URI="your_uri" -a lyrics-generator-app

# 4. Deploy
git push heroku main

# 5. View logs
heroku logs --tail -a lyrics-generator-app
```

## Security Considerations

### Environment Variables

```bash
# Never commit sensitive data
# Use environment variables for:
MONGODB_URI="secure_connection_string"
API_KEYS="secret_keys"
DOCKERHUB_TOKEN="token"

# Use .env.example for templates
# Add .env to .gitignore
```

### Network Security

```bash
# Use HTTPS in production
# Configure firewall rules
# Restrict access to only needed ports
# Use VPN for internal services
```

### Container Security

```dockerfile
# In Dockerfile
# 1. Use specific base image version
FROM python:3.11-slim

# 2. Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# 3. Scan for vulnerabilities
RUN pip install safety
RUN safety check
```

Scan with Trivy:
```bash
trivy image your-registry/lyrics-generator:latest
```

### Data Protection

```bash
# Encrypt sensitive data in transit
# Use TLS/SSL certificates
# Implement authentication/authorization
# Regular security audits
```

## Monitoring and Logging

### Application Logging

```python
# In main.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Application started")
logger.error("Error occurred", exc_info=True)
```

### Docker Logs

```bash
# View logs
docker logs lyrics-app

# Follow logs
docker logs -f lyrics-app

# Last 100 lines
docker logs --tail 100 lyrics-app
```

### Health Checks

```bash
# Check application health
curl http://localhost:8502

# Monitor with Prometheus (optional)
# Monitor with ELK Stack (optional)
# Monitor with Datadog (optional)
```

## Scaling

### Horizontal Scaling (Docker)

```bash
# Run multiple instances
docker run -d -p 8503:8502 lyrics-generator:latest
docker run -d -p 8504:8502 lyrics-generator:latest

# Use load balancer (Nginx, HAProxy)
```

### Kubernetes Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lyrics-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lyrics-generator
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Troubleshooting

### Common Deployment Issues

**Container won't start**
```bash
# Check logs
docker logs container-name

# Check image exists
docker images | grep lyrics

# Rebuild image
docker build --no-cache -t lyrics-generator:latest .
```

**Port binding error**
```bash
# Find process using port
lsof -i :8502

# Kill process
kill -9 <PID>
```

**MongoDB connection error**
```bash
# Test connection
mongo "mongodb+srv://user:pass@cluster.mongodb.net"

# Check environment variables
echo $MONGODB_URI
```

**Memory issues**
```bash
# Increase memory limit
docker run -m 2g lyrics-generator:latest

# For Kubernetes
# Update resource limits in deployment.yaml
```

**Performance degradation**
```bash
# Check resource usage
docker stats

# Profile application
# Use APM tools (DataDog, New Relic, etc.)

# Optimize model loading (caching)
# Use GPU acceleration if available
```

## Post-Deployment

### Monitoring

- Set up log aggregation (ELK, CloudWatch)
- Configure alerting (errors, crashes)
- Monitor resource usage
- Track response times

### Maintenance

- Regular security updates
- Dependency updates
- Database backups
- Disaster recovery planning

### Documentation

- Keep deployment docs updated
- Document configurations
- Create runbooks for common issues
- Train team on processes

## Support

- Documentation: [README.md](README.md)
- Installation: [INSTALLATION.md](INSTALLATION.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Issues: https://github.com/Mayankvlog/lyrics_generator_generative_ai/issues
- Email: mayankkr0311@gmail.com

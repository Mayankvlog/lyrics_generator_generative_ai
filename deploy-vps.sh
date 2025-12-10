#!/bin/bash
# VPS Deployment Script for Lyrics Generator

set -e

echo "📦 Updating system packages..."
apt-get update
apt-get upgrade -y

echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker root
fi

echo "🔧 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

echo "📂 Cloning repository..."
cd /root
if [ -d "lyrics_generator_generative_ai" ]; then
    cd lyrics_generator_generative_ai
    git pull origin main
else
    git clone https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
    cd lyrics_generator_generative_ai
fi

echo "📝 Creating .env file..."
cat > .env << EOF
MONGO_URI=mongodb://localhost:27017/lyrics
VPS_HOST=167.71.235.91
VPS_USER=root
VPS_PASSWORD=${VPS_PASSWORD:-your_password}
DOCKERHUB_USERNAME=mayank035
DOCKERHUB_TOKEN=${DOCKERHUB_TOKEN:-your_token}
EOF

echo "🚀 Starting Docker containers..."
docker-compose up -d

echo "✅ Deployment complete!"
echo "🌐 Access your app at: http://167.71.235.91:8501"
echo "📊 View logs: docker-compose logs -f app"

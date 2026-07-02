#!/bin/bash
# AutoML Web Platform - EC2 Setup Script
# Run as root or with sudo on Ubuntu 22.04+

set -e

echo "=== AutoML Platform - EC2 Setup ==="

# Update system
echo "Updating system packages..."
apt-get update && apt-get upgrade -y

# Install dependencies
echo "Installing system dependencies..."
apt-get install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx git

# Create app directory
APP_DIR="/home/ubuntu/AutoML"
if [ ! -d "$APP_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/SriniPamujula/WebAnalytics.git "$APP_DIR"
fi

cd "$APP_DIR"

# Create virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create required directories
mkdir -p uploads static/generated

# Set ownership
chown -R ubuntu:ubuntu "$APP_DIR"

# Install systemd service
echo "Configuring systemd service..."
cp deploy/automl.service /etc/systemd/system/automl.service
systemctl daemon-reload
systemctl enable automl
systemctl start automl

# Configure nginx
echo "Configuring nginx..."
cp deploy/automl-nginx.conf /etc/nginx/sites-available/automl
ln -sf /etc/nginx/sites-available/automl /etc/nginx/sites-enabled/automl
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# SSL Certificate
echo "Obtaining SSL certificate..."
echo "Make sure DNS A record for automl.betterai.guru points to this server's IP first."
echo "Then run: sudo certbot --nginx -d automl.betterai.guru"

echo ""
echo "=== Setup Complete ==="
echo "1. Point DNS: automl.betterai.guru -> $(curl -s http://checkip.amazonaws.com)"
echo "2. Run: sudo certbot --nginx -d automl.betterai.guru"
echo "3. Test: https://automl.betterai.guru"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status automl    # Check app status"
echo "  sudo systemctl restart automl   # Restart app"
echo "  sudo journalctl -u automl -f    # View logs"

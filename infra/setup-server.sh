#!/bin/bash
# ============================================================
# One-time server setup script
# Run this ONCE on the server before the first deployment.
# Usage: bash setup-server.sh
# ============================================================
set -e

echo "============================================"
echo "  Server Setup for OCR Project + Caddy"
echo "============================================"

# --- 1. Create caddy directory ---
CADDY_DIR="$HOME/caddy"
if [ ! -d "$CADDY_DIR" ]; then
    echo "📂 Creating Caddy directory at $CADDY_DIR..."
    mkdir -p "$CADDY_DIR"
else
    echo "📂 Caddy directory already exists at $CADDY_DIR"
fi

# --- 2. Copy caddy config ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "📋 Copying Caddy config..."
cp "$SCRIPT_DIR/caddy/docker-compose.yml" "$CADDY_DIR/"

# --- 3. Start Caddy ---
echo "🚀 Starting Caddy..."
cd "$CADDY_DIR"
docker compose up -d

echo ""
echo "✅ Caddy is running! HTTPS is automatic."
echo ""
echo "Next steps:"
echo "  1. Make sure your domain (webdev2.vayupak.net) points to this server's IP"
echo "  2. Clone your project and run: docker compose up -d"
echo ""

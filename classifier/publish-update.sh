#!/bin/bash
set -e

echo "🚀 Publishing water-conflict-classifier update"
echo "=============================================="

# Get latest version from PyPI
echo "🔍 Fetching latest version from PyPI..."
PYPI_VERSION=$(curl -s https://pypi.org/pypi/water-conflict-classifier/json | python3 -c "import sys, json; print(json.load(sys.stdin)['info']['version'])")

if [ -z "$PYPI_VERSION" ]; then
    echo "❌ Failed to fetch version from PyPI"
    exit 1
fi

echo "📦 Latest PyPI version: $PYPI_VERSION"

# Auto-increment patch version (e.g., 0.1.1 -> 0.1.2)
IFS='.' read -ra VERSION_PARTS <<< "$PYPI_VERSION"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"

echo "📈 New version: $NEW_VERSION"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted"
    exit 1
fi

# Update version in pyproject.toml
echo "✏️  Updating version in pyproject.toml..."
sed -i '' "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml

# Update version in train_on_hf.py
echo "✏️  Updating version in ../scripts/train_on_hf.py..."
sed -i '' "s/water-conflict-classifier>=.*/water-conflict-classifier>=$NEW_VERSION\",/" ../scripts/train_on_hf.py

# Clean old builds
echo "🧹 Cleaning old builds..."
rm -rf dist/ build/ *.egg-info

# Build
echo "🔨 Building package..."
uv build

# Publish
echo "📤 Publishing to PyPI..."
uv publish

echo ""
echo "✅ Published version $NEW_VERSION!"
echo "🔗 https://pypi.org/project/water-conflict-classifier/$NEW_VERSION/"


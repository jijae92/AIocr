#!/bin/bash
# Automated build script for macOS .app bundle
# Usage:
#   ./scripts/build_macos.sh                    # Build only
#   CODESIGN_IDENTITY="..." ./scripts/build_macos.sh    # Build and sign
#   CODESIGN_IDENTITY="..." NOTARY_PROFILE="..." ./scripts/build_macos.sh  # Build, sign, and notarize

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Building Hybrid PDF OCR for macOS...${NC}"
echo ""

# Configuration
APP_NAME="HybridPDFOCR"
VERSION="0.1.0"
BUNDLE_ID="com.hybridpdfocr.app"

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC_FILE="$PROJECT_ROOT/app_macos.spec"
ENTITLEMENTS_FILE="$PROJECT_ROOT/entitlements.plist"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"
APP_PATH="$DIST_DIR/${APP_NAME}.app"
ZIP_PATH="$DIST_DIR/${APP_NAME}.zip"
RELEASE_ZIP="${APP_NAME}-v${VERSION}.zip"

cd "$PROJECT_ROOT"

# Step 1: Clean previous builds
echo -e "${YELLOW}🧹 Cleaning previous builds...${NC}"
rm -rf "$BUILD_DIR" "$DIST_DIR"
rm -f "$ZIP_PATH" "$RELEASE_ZIP"

# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# Step 2: Verify dependencies
echo -e "${YELLOW}📋 Checking dependencies...${NC}"

if ! command -v pyinstaller &> /dev/null; then
    echo -e "${RED}✗ PyInstaller not found. Install with: pip install pyinstaller${NC}"
    exit 1
fi

echo -e "${GREEN}✓ PyInstaller found: $(pyinstaller --version)${NC}"

# Check if spec file exists
if [ ! -f "$SPEC_FILE" ]; then
    echo -e "${RED}✗ Spec file not found: $SPEC_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Spec file found${NC}"
echo ""

# Step 3: Build the app bundle
echo -e "${YELLOW}📦 Building app bundle...${NC}"
pyinstaller "$SPEC_FILE" --clean --noconfirm

if [ ! -d "$APP_PATH" ]; then
    echo -e "${RED}✗ Build failed: $APP_PATH not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ App bundle built successfully${NC}"
echo ""

# Step 4: Show bundle size
BUNDLE_SIZE=$(du -sh "$APP_PATH" | cut -f1)
echo -e "${BLUE}📦 Bundle size: $BUNDLE_SIZE${NC}"
echo ""

# Step 5: Code signing (optional)
if [ -n "$CODESIGN_IDENTITY" ]; then
    echo -e "${YELLOW}✍️  Signing app bundle...${NC}"

    # Check if entitlements file exists
    if [ ! -f "$ENTITLEMENTS_FILE" ]; then
        echo -e "${RED}✗ Entitlements file not found: $ENTITLEMENTS_FILE${NC}"
        exit 1
    fi

    # Sign the app
    codesign --force --deep \
        --sign "$CODESIGN_IDENTITY" \
        --options runtime \
        --entitlements "$ENTITLEMENTS_FILE" \
        --timestamp \
        "$APP_PATH"

    echo -e "${GREEN}✓ App signed with: $CODESIGN_IDENTITY${NC}"

    # Verify signature
    echo -e "${YELLOW}🔍 Verifying signature...${NC}"
    if codesign --verify --deep --strict --verbose=2 "$APP_PATH" 2>&1; then
        echo -e "${GREEN}✓ Signature verified${NC}"
    else
        echo -e "${RED}✗ Signature verification failed${NC}"
        exit 1
    fi

    # Check Gatekeeper
    echo -e "${YELLOW}🔍 Checking Gatekeeper compatibility...${NC}"
    if spctl --assess --type execute --verbose "$APP_PATH" 2>&1; then
        echo -e "${GREEN}✓ Gatekeeper assessment passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Gatekeeper assessment failed (expected if not notarized)${NC}"
    fi

    echo ""
else
    echo -e "${YELLOW}⚠️  Skipping code signing (CODESIGN_IDENTITY not set)${NC}"
    echo -e "${YELLOW}   For local testing, run: xattr -cr $APP_PATH${NC}"
    echo ""
fi

# Step 6: Notarization (optional)
if [ -n "$NOTARY_PROFILE" ] && [ -n "$CODESIGN_IDENTITY" ]; then
    echo -e "${YELLOW}📝 Creating ZIP for notarization...${NC}"
    ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"
    echo -e "${GREEN}✓ ZIP created: $ZIP_PATH${NC}"

    echo -e "${YELLOW}🔐 Submitting for notarization...${NC}"
    echo -e "${BLUE}   This may take 2-15 minutes...${NC}"

    if xcrun notarytool submit "$ZIP_PATH" \
        --keychain-profile "$NOTARY_PROFILE" \
        --wait 2>&1; then
        echo -e "${GREEN}✓ Notarization successful${NC}"

        echo -e "${YELLOW}📎 Stapling notarization ticket...${NC}"
        xcrun stapler staple "$APP_PATH"
        echo -e "${GREEN}✓ Notarization ticket stapled${NC}"

        echo -e "${YELLOW}🔍 Verifying notarization...${NC}"
        if xcrun stapler validate "$APP_PATH" 2>&1; then
            echo -e "${GREEN}✓ Notarization verified${NC}"
        else
            echo -e "${RED}✗ Notarization verification failed${NC}"
            exit 1
        fi

        # Check Gatekeeper again
        if spctl --assess --type install --verbose "$APP_PATH" 2>&1; then
            echo -e "${GREEN}✓ Gatekeeper assessment passed (notarized)${NC}"
        fi
    else
        echo -e "${RED}✗ Notarization failed${NC}"
        echo -e "${YELLOW}   Check logs with: xcrun notarytool log <submission-id> --keychain-profile \"$NOTARY_PROFILE\"${NC}"
        exit 1
    fi

    # Remove temporary ZIP
    rm -f "$ZIP_PATH"
    echo ""
else
    if [ -n "$CODESIGN_IDENTITY" ]; then
        echo -e "${YELLOW}⚠️  Skipping notarization (NOTARY_PROFILE not set)${NC}"
    else
        echo -e "${YELLOW}⚠️  Skipping notarization (signing required first)${NC}"
    fi
    echo ""
fi

# Step 7: Create distribution ZIP
echo -e "${YELLOW}📦 Creating distribution ZIP...${NC}"
ditto -c -k --keepParent "$APP_PATH" "$RELEASE_ZIP"

if [ -f "$RELEASE_ZIP" ]; then
    RELEASE_SIZE=$(du -sh "$RELEASE_ZIP" | cut -f1)
    echo -e "${GREEN}✓ Distribution ZIP created: $RELEASE_ZIP ($RELEASE_SIZE)${NC}"
else
    echo -e "${RED}✗ Failed to create distribution ZIP${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo -e "${BLUE}📍 Outputs:${NC}"
echo -e "   App bundle:      ${APP_PATH}"
echo -e "   Distribution:    ${RELEASE_ZIP}"
echo ""

# Print usage instructions
if [ -z "$CODESIGN_IDENTITY" ]; then
    echo -e "${YELLOW}ℹ️  For local testing:${NC}"
    echo -e "   1. Remove quarantine: ${BLUE}xattr -cr $APP_PATH${NC}"
    echo -e "   2. Run app: ${BLUE}open $APP_PATH${NC}"
    echo ""
    echo -e "${YELLOW}ℹ️  For distribution:${NC}"
    echo -e "   Rebuild with code signing:"
    echo -e "   ${BLUE}CODESIGN_IDENTITY=\"Developer ID Application: Your Name (TEAM_ID)\" ./scripts/build_macos.sh${NC}"
elif [ -z "$NOTARY_PROFILE" ]; then
    echo -e "${YELLOW}ℹ️  For full distribution:${NC}"
    echo -e "   Rebuild with notarization:"
    echo -e "   ${BLUE}CODESIGN_IDENTITY=\"$CODESIGN_IDENTITY\" NOTARY_PROFILE=\"notary-profile\" ./scripts/build_macos.sh${NC}"
else
    echo -e "${GREEN}🎉 App is signed and notarized!${NC}"
    echo -e "   Ready for distribution: ${BLUE}$RELEASE_ZIP${NC}"
fi

echo ""

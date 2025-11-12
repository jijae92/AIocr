# macOS Packaging Guide

Complete guide for packaging Hybrid PDF OCR as a macOS .app bundle with codesigning and notarization.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building the .app Bundle](#building-the-app-bundle)
3. [Testing Locally](#testing-locally)
4. [Code Signing](#code-signing)
5. [Notarization](#notarization)
6. [Distribution](#distribution)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

```bash
# Install PyInstaller
pip install pyinstaller

# Verify installation
pyinstaller --version
```

### Optional: Create App Icon

Create `resources/icon.icns` from a PNG image:

```bash
# Install iconutil (included with Xcode)
mkdir -p resources/icon.iconset

# Create multiple sizes (16x16 to 512x512)
sips -z 16 16 icon.png --out resources/icon.iconset/icon_16x16.png
sips -z 32 32 icon.png --out resources/icon.iconset/icon_16x16@2x.png
sips -z 32 32 icon.png --out resources/icon.iconset/icon_32x32.png
sips -z 64 64 icon.png --out resources/icon.iconset/icon_32x32@2x.png
sips -z 128 128 icon.png --out resources/icon.iconset/icon_128x128.png
sips -z 256 256 icon.png --out resources/icon.iconset/icon_128x128@2x.png
sips -z 256 256 icon.png --out resources/icon.iconset/icon_256x256.png
sips -z 512 512 icon.png --out resources/icon.iconset/icon_256x256@2x.png
sips -z 512 512 icon.png --out resources/icon.iconset/icon_512x512.png
sips -z 1024 1024 icon.png --out resources/icon.iconset/icon_512x512@2x.png

# Convert to .icns
iconutil -c icns resources/icon.iconset -o resources/icon.icns
```

---

## Building the .app Bundle

### 1. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build/ dist/

# Remove PyInstaller cache
rm -rf __pycache__ src/**/__pycache__
```

### 2. Build the App

```bash
# Build using spec file
pyinstaller app_macos.spec

# Output will be at: dist/HybridPDFOCR.app
```

### 3. Verify Bundle Structure

```bash
# Check bundle structure
ls -la dist/HybridPDFOCR.app/Contents/

# Expected structure:
# MacOS/          - Executable
# Resources/      - Icon and data files
# Frameworks/     - Dynamic libraries
# Info.plist      - Bundle metadata
```

### 4. Check Bundle Size

```bash
du -sh dist/HybridPDFOCR.app
```

**Note:** Initial bundle may be large (500MB-2GB) due to PyTorch, Transformers, and other ML dependencies.

**Size Optimization Tips:**
- Use ONNX models instead of PyTorch models
- Exclude unused ML frameworks
- Use `upx` compression (already enabled in spec)
- Strip debug symbols

---

## Testing Locally

### 1. Run the App

```bash
# Run from command line (see console output)
./dist/HybridPDFOCR.app/Contents/MacOS/HybridPDFOCR

# Or double-click in Finder
open dist/HybridPDFOCR.app
```

### 2. Check for Errors

If the app doesn't launch:

```bash
# View console logs
log stream --predicate 'process == "HybridPDFOCR"' --level debug

# Or check system logs
tail -f /var/log/system.log | grep HybridPDFOCR
```

### 3. Verify Features

- [ ] PDF file loading
- [ ] OCR processing (with available engines)
- [ ] Export to searchable PDF
- [ ] Configuration loading
- [ ] GUI rendering

---

## Code Signing

Code signing is **required** for distribution outside the App Store and for Gatekeeper compatibility.

### 1. Prerequisites

- **Apple Developer Account** ($99/year)
- **Developer ID Application Certificate**

Get certificate from [Apple Developer Portal](https://developer.apple.com/account/resources/certificates/list):
1. Certificates, Identifiers & Profiles
2. Certificates → Create → Developer ID Application
3. Download and install in Keychain Access

### 2. List Available Identities

```bash
# Find your Developer ID
security find-identity -v -p codesigning

# Output example:
# 1) ABCD1234... "Developer ID Application: Your Name (TEAM_ID)"
```

### 3. Sign the App

```bash
# Set your identity
CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"

# Sign with hardened runtime
codesign --force --deep --sign "$CODESIGN_IDENTITY" \
  --options runtime \
  --entitlements entitlements.plist \
  --timestamp \
  dist/HybridPDFOCR.app
```

### 4. Create Entitlements File

Create `entitlements.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- Allow JIT compilation (required for PyTorch) -->
    <key>com.apple.security.cs.allow-jit</key>
    <true/>

    <!-- Allow unsigned executable memory -->
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>

    <!-- Disable library validation (required for pip-installed libraries) -->
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>

    <!-- Allow DYLD environment variables -->
    <key>com.apple.security.cs.allow-dyld-environment-variables</key>
    <true/>

    <!-- Network access (for DocAI, model downloads) -->
    <key>com.apple.security.network.client</key>
    <true/>

    <key>com.apple.security.network.server</key>
    <true/>

    <!-- File access -->
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>

    <key>com.apple.security.files.downloads.read-write</key>
    <true/>
</dict>
</plist>
```

### 5. Verify Signature

```bash
# Check signature
codesign --verify --deep --strict --verbose=2 dist/HybridPDFOCR.app

# Display signature info
codesign -dvv dist/HybridPDFOCR.app

# Expected output includes:
# - Identifier: com.hybridpdfocr.app
# - Authority: Developer ID Application: Your Name
# - Timestamp: (timestamp server)
# - Runtime Version: 10.15.0 (or higher)
```

### 6. Test Gatekeeper

```bash
# Test if macOS will allow the app to run
spctl --assess --type execute --verbose dist/HybridPDFOCR.app

# Expected: "accepted"
```

---

## Notarization

Notarization is **required** for macOS 10.15+ to avoid Gatekeeper warnings.

### 1. Prerequisites

- Signed app bundle
- **App-specific password** for Apple ID

Generate app-specific password:
1. Go to [appleid.apple.com](https://appleid.apple.com)
2. Sign in → Security → App-Specific Passwords
3. Generate password and save it

### 2. Create ZIP Archive

```bash
# Notarization requires a ZIP, DMG, or PKG
ditto -c -k --keepParent dist/HybridPDFOCR.app dist/HybridPDFOCR.zip
```

### 3. Submit for Notarization

```bash
# Store credentials (first time only)
xcrun notarytool store-credentials "notary-profile" \
  --apple-id "your-apple-id@example.com" \
  --team-id "YOUR_TEAM_ID" \
  --password "xxxx-xxxx-xxxx-xxxx"  # App-specific password

# Submit for notarization
xcrun notarytool submit dist/HybridPDFOCR.zip \
  --keychain-profile "notary-profile" \
  --wait

# Output:
# Submission ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# Status: Accepted (or In Progress)
```

**Note:** Notarization typically takes 2-15 minutes.

### 4. Check Notarization Status

```bash
# Check status (if not using --wait)
xcrun notarytool info <submission-id> \
  --keychain-profile "notary-profile"

# View detailed log
xcrun notarytool log <submission-id> \
  --keychain-profile "notary-profile"
```

### 5. Staple the Ticket

After successful notarization:

```bash
# Staple notarization ticket to app
xcrun stapler staple dist/HybridPDFOCR.app

# Verify stapling
xcrun stapler validate dist/HybridPDFOCR.app

# Expected: "The validate action worked!"
```

### 6. Verify Notarization

```bash
# Check if notarized
spctl --assess -vv --type install dist/HybridPDFOCR.app

# Expected: "accepted" with "source=Notarized Developer ID"
```

---

## Distribution

### Option 1: Direct Download (ZIP)

```bash
# Create distribution ZIP
ditto -c -k --keepParent dist/HybridPDFOCR.app HybridPDFOCR-v0.1.0.zip

# Upload to website/GitHub Releases
```

**User Installation:**
1. Download ZIP
2. Extract
3. Drag `HybridPDFOCR.app` to `/Applications`
4. Double-click to launch

### Option 2: DMG Installer

Create a DMG for more professional distribution:

```bash
# Install create-dmg
brew install create-dmg

# Create DMG
create-dmg \
  --volname "Hybrid PDF OCR" \
  --volicon "resources/icon.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "HybridPDFOCR.app" 200 190 \
  --hide-extension "HybridPDFOCR.app" \
  --app-drop-link 600 185 \
  "HybridPDFOCR-v0.1.0.dmg" \
  "dist/HybridPDFOCR.app"

# Sign the DMG (optional but recommended)
codesign --sign "$CODESIGN_IDENTITY" HybridPDFOCR-v0.1.0.dmg

# Notarize the DMG
xcrun notarytool submit HybridPDFOCR-v0.1.0.dmg \
  --keychain-profile "notary-profile" \
  --wait

# Staple
xcrun stapler staple HybridPDFOCR-v0.1.0.dmg
```

### Option 3: Mac App Store

For App Store distribution:
1. Create App Store Distribution certificate
2. Create provisioning profile
3. Build with App Store profile
4. Submit via Xcode or Transporter
5. See [App Store Connect Guide](https://developer.apple.com/app-store-connect/)

---

## Troubleshooting

### App Won't Launch

**Symptom:** App icon bounces and quits immediately

**Solutions:**

1. **Check console logs:**
   ```bash
   log stream --predicate 'process == "HybridPDFOCR"' --level debug
   ```

2. **Run from command line to see errors:**
   ```bash
   ./dist/HybridPDFOCR.app/Contents/MacOS/HybridPDFOCR
   ```

3. **Missing dependencies:** Check hiddenimports in `app_macos.spec`

4. **Python path issues:** Verify `sys.path` includes app bundle

### Gatekeeper Blocks App

**Symptom:** "HybridPDFOCR is damaged and can't be opened"

**Solutions:**

1. **Remove quarantine attribute (for local testing only):**
   ```bash
   xattr -cr dist/HybridPDFOCR.app
   ```

2. **Sign the app properly** (see Code Signing section)

3. **Notarize the app** (required for distribution)

### Import Errors

**Symptom:** `ModuleNotFoundError` when running app

**Solutions:**

1. **Add missing module to hiddenimports:**
   ```python
   # In app_macos.spec
   hiddenimports = [
       'missing_module',
       # ...
   ]
   ```

2. **Rebuild:**
   ```bash
   pyinstaller app_macos.spec
   ```

### Large Bundle Size

**Solutions:**

1. **Exclude unused packages:**
   ```python
   # In app_macos.spec
   excludes = [
       'matplotlib',
       'jupyter',
       'pytest',
       # Add more
   ]
   ```

2. **Use ONNX instead of PyTorch:**
   - Export models to ONNX
   - Remove PyTorch from dependencies

3. **Strip unnecessary files:**
   ```python
   # In app_macos.spec, after Analysis
   a.datas = [x for x in a.datas if not x[0].startswith('tests/')]
   ```

### Notarization Rejected

**Common reasons:**

1. **Hardened runtime not enabled**
   ```bash
   codesign --options runtime ...
   ```

2. **Missing entitlements**
   - Add required entitlements to `entitlements.plist`

3. **Unsigned components**
   ```bash
   # Sign with --deep flag
   codesign --deep --sign "$CODESIGN_IDENTITY" ...
   ```

4. **Check rejection log:**
   ```bash
   xcrun notarytool log <submission-id> --keychain-profile "notary-profile"
   ```

---

## Automated Build Script

Create `scripts/build_macos.sh`:

```bash
#!/bin/bash
# Automated build script for macOS .app bundle

set -e  # Exit on error

echo "🚀 Building Hybrid PDF OCR for macOS..."

# Clean
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/

# Build
echo "📦 Building app bundle..."
pyinstaller app_macos.spec

# Sign (if CODESIGN_IDENTITY is set)
if [ -n "$CODESIGN_IDENTITY" ]; then
    echo "✍️  Signing app bundle..."
    codesign --force --deep --sign "$CODESIGN_IDENTITY" \
      --options runtime \
      --entitlements entitlements.plist \
      --timestamp \
      dist/HybridPDFOCR.app

    echo "✅ Verifying signature..."
    codesign --verify --deep --strict --verbose=2 dist/HybridPDFOCR.app
fi

# Notarize (if NOTARY_PROFILE is set)
if [ -n "$NOTARY_PROFILE" ]; then
    echo "📝 Creating ZIP for notarization..."
    ditto -c -k --keepParent dist/HybridPDFOCR.app dist/HybridPDFOCR.zip

    echo "🔐 Submitting for notarization..."
    xcrun notarytool submit dist/HybridPDFOCR.zip \
      --keychain-profile "$NOTARY_PROFILE" \
      --wait

    echo "📎 Stapling notarization ticket..."
    xcrun stapler staple dist/HybridPDFOCR.app

    echo "✅ Verifying notarization..."
    xcrun stapler validate dist/HybridPDFOCR.app
fi

# Create distribution ZIP
echo "📦 Creating distribution ZIP..."
ditto -c -k --keepParent dist/HybridPDFOCR.app HybridPDFOCR-v0.1.0.zip

echo "✅ Build complete!"
echo "📍 App location: dist/HybridPDFOCR.app"
echo "📍 Distribution ZIP: HybridPDFOCR-v0.1.0.zip"

# Show bundle size
du -sh dist/HybridPDFOCR.app
```

**Usage:**

```bash
# Make executable
chmod +x scripts/build_macos.sh

# Build only (no signing)
./scripts/build_macos.sh

# Build and sign
CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAM_ID)" \
./scripts/build_macos.sh

# Build, sign, and notarize
CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAM_ID)" \
NOTARY_PROFILE="notary-profile" \
./scripts/build_macos.sh
```

---

## References

- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [Apple Code Signing Guide](https://developer.apple.com/documentation/security/code_signing_services)
- [Apple Notarization Guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [Hardened Runtime](https://developer.apple.com/documentation/security/hardened_runtime)
- [Gatekeeper](https://support.apple.com/en-us/HT202491)

---

## Summary

### For Local Development (No Signing)

```bash
# Build
pyinstaller app_macos.spec

# Remove quarantine
xattr -cr dist/HybridPDFOCR.app

# Run
open dist/HybridPDFOCR.app
```

### For Distribution (With Signing & Notarization)

```bash
# 1. Build
pyinstaller app_macos.spec

# 2. Sign
codesign --force --deep --sign "$CODESIGN_IDENTITY" \
  --options runtime --entitlements entitlements.plist \
  --timestamp dist/HybridPDFOCR.app

# 3. Create ZIP
ditto -c -k --keepParent dist/HybridPDFOCR.app dist/HybridPDFOCR.zip

# 4. Notarize
xcrun notarytool submit dist/HybridPDFOCR.zip \
  --keychain-profile "notary-profile" --wait

# 5. Staple
xcrun stapler staple dist/HybridPDFOCR.app

# 6. Create distribution archive
ditto -c -k --keepParent dist/HybridPDFOCR.app HybridPDFOCR-v0.1.0.zip
```

That's it! Your Hybrid PDF OCR app is now ready for distribution on macOS. 🎉

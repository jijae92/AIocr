# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Hybrid PDF OCR macOS .app bundle.

Build with:
    pyinstaller app_macos.spec

Output:
    dist/HybridPDFOCR.app

Requirements:
    pip install pyinstaller
"""

import sys
from pathlib import Path

block_cipher = None

# Project root
root_dir = Path.cwd()
src_dir = root_dir / 'src'

# Data files to include
datas = [
    # Configuration files
    ('configs/*.yaml', 'configs'),

    # Model directories (empty placeholders, users download models separately)
    ('models/.gitkeep', 'models'),

    # Icon and resources
    # ('resources/icon.icns', 'resources'),  # Uncomment when icon is available
]

# Hidden imports (packages not detected by PyInstaller)
hiddenimports = [
    # PyQt5 modules
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.sip',

    # Transformers and dependencies
    'transformers',
    'transformers.models',
    'transformers.models.vision_encoder_decoder',
    'transformers.models.donut',
    'transformers.models.trocr',

    # ONNX Runtime
    'onnxruntime',
    'onnxruntime.capi',

    # Image processing
    'PIL',
    'PIL.Image',
    'cv2',

    # PDF processing
    'fitz',
    'pymupdf',
    'pdf2image',

    # ML frameworks
    'torch',
    'torch.nn',
    'torch.optim',
    'torchvision',

    # Accelerate and PEFT
    'accelerate',
    'peft',

    # Google Cloud
    'google.cloud',
    'google.cloud.documentai',
    'google.cloud.documentai_v1',
    'google.api_core',
    'google.auth',

    # Utilities
    'yaml',
    'jiwer',
    'tqdm',
    'numpy',
    'pandas',

    # Our modules
    'cache',
    'cache.store',
    'connectors',
    'connectors.docai_client',
    'engines',
    'engines.donut_engine',
    'engines.trocr_onnx',
    'engines.tatr_tables',
    'engines.pix2tex_math',
    'models',
    'data.ocr_result',
    'preproc',
    'preproc.pdf_loader',
    'preproc.vision_filters',
    'postproc',
    'postproc.text_norm',
    'postproc.layout_merge',
    'postproc.exporters',
    'router',
    'router.heuristics',
    'pdf',
    'pdf.text_layer',
    'eval',
    'eval.bench',
    'train',
    'train.donut_lora',
    'train.synthdog_loader',
    'train.export_trocr_onnx',
    'util',
    'util.config',
    'util.logging',
    'util.device',
    'util.coords',
    'util.timing',
]

# Binary exclusions (optional, to reduce size)
excludes = [
    'matplotlib',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'sphinx',
]

# Analysis
a = Analysis(
    ['src/gui/desktop_app.py'],
    pathex=[str(src_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary files to reduce size
# Remove test files
a.datas = [x for x in a.datas if not x[0].startswith('tests/')]

# Remove __pycache__
a.datas = [x for x in a.datas if '__pycache__' not in x[0]]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HybridPDFOCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # --windowed mode (no console)
    disable_windowed_traceback=False,
    # argv_emulation can cause the app script to be re-invoked
    # multiple times on macOS (especially when associated with
    # document types), which may manifest as multiple GUI windows
    # being opened repeatedly. Disable it to ensure a single launch.
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,  # Set for codesigning
    entitlements_file=None,  # Set for hardened runtime
    icon='resources/icon.icns' if (root_dir / 'resources/icon.icns').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HybridPDFOCR',
)

# macOS .app bundle
app = BUNDLE(
    coll,
    name='HybridPDFOCR.app',
    icon='resources/icon.icns' if (root_dir / 'resources/icon.icns').exists() else None,
    bundle_identifier='com.hybridpdfocr.localdev',
    version='0.1.0',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'CFBundleName': 'Hybrid PDF OCR',
        'CFBundleDisplayName': 'Hybrid PDF OCR',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'NSHumanReadableCopyright': 'Copyright © 2025',
        'NSHighResolutionCapable': True,

        # Document types (PDF support)
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'PDF Document',
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate',
                'LSItemContentTypes': ['com.adobe.pdf'],
                'CFBundleTypeExtensions': ['pdf'],
            }
        ],

        # Required for macOS 10.15+
        'NSRequiresAquaSystemAppearance': False,

        # Camera/photo library access (if needed for image import)
        'NSCameraUsageDescription': 'This app needs access to the camera to capture documents.',
        'NSPhotoLibraryUsageDescription': 'This app needs access to your photo library to import images.',

        # File access
        'NSDocumentsFolderUsageDescription': 'This app needs access to your documents to process PDF files.',
        'NSDownloadsFolderUsageDescription': 'This app needs access to your downloads to process PDF files.',

        # Prevent multiple instances of the .app from running
        # simultaneously. Combined with the Python-level lock in
        # src/gui/desktop_app.py this helps ensure only a single
        # window / process is active at once.
        'LSMultipleInstancesProhibited': True,
    },
)

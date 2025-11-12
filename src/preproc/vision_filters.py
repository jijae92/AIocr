"""
Vision preprocessing filters for document images.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage import exposure
from skimage.filters import threshold_otsu

from util.logging import get_logger

logger = get_logger(__name__)


class VisionFilters:
    """Document image preprocessing filters."""

    @staticmethod
    def to_grayscale(image: Image.Image) -> Image.Image:
        """
        Convert image to grayscale.

        Args:
            image: Input PIL image

        Returns:
            Grayscale PIL image
        """
        return image.convert('L')

    @staticmethod
    def enhance_contrast(
        image: Image.Image,
        method: str = 'clahe',
    ) -> Image.Image:
        """
        Enhance image contrast.

        Args:
            image: Input PIL image
            method: Enhancement method ('clahe', 'hist_eq', 'adaptive')

        Returns:
            Enhanced PIL image
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array

        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_gray)

        elif method == 'hist_eq':
            # Standard histogram equalization
            enhanced = cv2.equalizeHist(img_gray)

        elif method == 'adaptive':
            # Adaptive histogram equalization using scikit-image
            enhanced = exposure.equalize_adapthist(img_gray, clip_limit=0.03)
            enhanced = (enhanced * 255).astype(np.uint8)

        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")

        # Convert back to PIL image
        return Image.fromarray(enhanced)

    @staticmethod
    def denoise(
        image: Image.Image,
        method: str = 'bilateral',
        strength: int = 5,
    ) -> Image.Image:
        """
        Denoise image.

        Args:
            image: Input PIL image
            method: Denoising method ('bilateral', 'gaussian', 'median')
            strength: Denoising strength (kernel size)

        Returns:
            Denoised PIL image
        """
        img_array = np.array(image)

        if method == 'bilateral':
            # Bilateral filter (edge-preserving)
            denoised = cv2.bilateralFilter(img_array, strength, 75, 75)

        elif method == 'gaussian':
            # Gaussian blur
            denoised = cv2.GaussianBlur(img_array, (strength, strength), 0)

        elif method == 'median':
            # Median filter
            denoised = cv2.medianBlur(img_array, strength)

        else:
            raise ValueError(f"Unknown denoising method: {method}")

        return Image.fromarray(denoised)

    @staticmethod
    def binarize(
        image: Image.Image,
        threshold: Optional[int] = None,
        method: str = 'otsu',
    ) -> Image.Image:
        """
        Binarize image (convert to black and white).

        Args:
            image: Input PIL image
            threshold: Manual threshold (0-255), None for automatic
            method: Thresholding method ('otsu', 'adaptive')

        Returns:
            Binary PIL image
        """
        # Convert to grayscale
        img_gray = np.array(image.convert('L'))

        if threshold is not None:
            # Manual threshold
            _, binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

        elif method == 'otsu':
            # Otsu's thresholding
            threshold_value = threshold_otsu(img_gray)
            binary = (img_gray > threshold_value).astype(np.uint8) * 255

        elif method == 'adaptive':
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                img_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )

        else:
            raise ValueError(f"Unknown binarization method: {method}")

        return Image.fromarray(binary)

    @staticmethod
    def deskew(image: Image.Image, max_angle: float = 10.0) -> Tuple[Image.Image, float]:
        """
        Deskew (straighten) image.

        Args:
            image: Input PIL image
            max_angle: Maximum angle to correct (degrees)

        Returns:
            Tuple of (deskewed image, detected angle)
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is None:
            logger.debug("No lines detected for deskewing")
            return image, 0.0

        # Calculate angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.rad2deg(theta) - 90
            # Filter out angles outside the expected range
            if abs(angle) <= max_angle:
                angles.append(angle)

        if not angles:
            logger.debug("No valid angles detected for deskewing")
            return image, 0.0

        # Use median angle
        median_angle = np.median(angles)

        # Rotate image
        if abs(median_angle) < 0.1:
            # Skip rotation if angle is negligible
            return image, 0.0

        h, w = img_array.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img_array, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)

        logger.debug(f"Deskewed image by {median_angle:.2f} degrees")
        return Image.fromarray(rotated), median_angle

    @staticmethod
    def auto_rotate(image: Image.Image) -> Tuple[Image.Image, int]:
        """
        Auto-rotate image based on orientation detection.

        Args:
            image: Input PIL image

        Returns:
            Tuple of (rotated image, rotation angle in degrees)
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Detect text regions using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.debug("No contours detected for auto-rotation")
            return image, 0

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]

        # Determine rotation angle (0, 90, 180, 270)
        if angle < -45:
            rotation_angle = 90
        elif angle > 45:
            rotation_angle = 270
        else:
            rotation_angle = 0

        if rotation_angle == 0:
            return image, 0

        # Rotate image
        rotated = image.rotate(-rotation_angle, expand=True)
        logger.debug(f"Auto-rotated image by {rotation_angle} degrees")

        return rotated, rotation_angle

    @staticmethod
    def remove_borders(
        image: Image.Image,
        threshold: int = 200,
    ) -> Image.Image:
        """
        Remove white borders from image.

        Args:
            image: Input PIL image
            threshold: Threshold for white detection (0-255)

        Returns:
            Cropped PIL image
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Threshold to binary
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Get bounding box of all contours
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Crop image
        if x_min < x_max and y_min < y_max:
            cropped = img_array[int(y_min) : int(y_max), int(x_min) : int(x_max)]
            return Image.fromarray(cropped)

        return image

    @staticmethod
    def preprocess_document(
        image: Image.Image,
        auto_rotate: bool = True,
        deskew: bool = True,
        enhance_contrast: bool = True,
        denoise: bool = False,
        binarize: bool = False,
    ) -> Image.Image:
        """
        Apply standard document preprocessing pipeline.

        Args:
            image: Input PIL image
            auto_rotate: Apply auto-rotation
            deskew: Apply deskewing
            enhance_contrast: Enhance contrast
            denoise: Apply denoising
            binarize: Apply binarization

        Returns:
            Preprocessed PIL image
        """
        processed = image

        # Auto-rotate
        if auto_rotate:
            processed, _ = VisionFilters.auto_rotate(processed)

        # Deskew
        if deskew:
            processed, _ = VisionFilters.deskew(processed)

        # Enhance contrast
        if enhance_contrast:
            processed = VisionFilters.enhance_contrast(processed, method='clahe')

        # Denoise
        if denoise:
            processed = VisionFilters.denoise(processed, method='bilateral')

        # Binarize
        if binarize:
            processed = VisionFilters.binarize(processed, method='otsu')

        return processed

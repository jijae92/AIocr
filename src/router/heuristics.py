"""
Routing and ensemble logic for hybrid OCR.
"""

from typing import Dict, List, Literal, Optional

from util.logging import get_logger

logger = get_logger(__name__)

RoutingStrategy = Literal['docai_only', 'hybrid', 'donut_primary', 'trocr_primary']
EnsembleMethod = Literal['max_confidence', 'voting', 'weighted_average']


class OCRRouter:
    """Router for selecting and combining OCR engines."""

    def __init__(
        self,
        strategy: RoutingStrategy = 'hybrid',
        docai_accept_threshold: float = 0.85,
        docai_fallback_threshold: float = 0.70,
        ensemble_range: tuple = (0.70, 0.85),
        ensemble_method: EnsembleMethod = 'weighted_average',
        ensemble_weights: Optional[List[float]] = None,
    ):
        """
        Initialize OCR router.

        Args:
            strategy: Routing strategy
            docai_accept_threshold: Accept DocAI if confidence above this
            docai_fallback_threshold: Use fallback if confidence below this
            ensemble_range: Confidence range for ensemble
            ensemble_method: Method for combining results
            ensemble_weights: Weights for weighted ensemble [docai, local]
        """
        self.strategy = strategy
        self.docai_accept_threshold = docai_accept_threshold
        self.docai_fallback_threshold = docai_fallback_threshold
        self.ensemble_range = ensemble_range
        self.ensemble_method = ensemble_method
        self.ensemble_weights = ensemble_weights or [0.6, 0.4]

        logger.info(f"Initialized OCR router with strategy: {strategy}")

    def route(
        self,
        docai_result: Optional[Dict] = None,
        local_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Route and combine OCR results.

        Args:
            docai_result: Result from Document AI
            local_result: Result from local model (Donut/TrOCR)

        Returns:
            Final OCR result
        """
        if self.strategy == 'docai_only':
            if docai_result is None:
                raise ValueError("DocAI result required for docai_only strategy")
            return self._select_result(docai_result, 'document_ai')

        elif self.strategy == 'donut_primary':
            if local_result is None:
                raise ValueError("Local result required for donut_primary strategy")
            return self._select_result(local_result, 'donut')

        elif self.strategy == 'trocr_primary':
            if local_result is None:
                raise ValueError("Local result required for trocr_primary strategy")
            return self._select_result(local_result, 'trocr')

        elif self.strategy == 'hybrid':
            return self._hybrid_route(docai_result, local_result)

        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")

    def _hybrid_route(
        self,
        docai_result: Optional[Dict],
        local_result: Optional[Dict],
    ) -> Dict:
        """
        Hybrid routing logic.

        Args:
            docai_result: DocAI result
            local_result: Local model result

        Returns:
            Selected or ensembled result
        """
        # If only one result available, use it
        if docai_result is None:
            if local_result is None:
                raise ValueError("At least one result required")
            return self._select_result(local_result, 'local')

        if local_result is None:
            return self._select_result(docai_result, 'document_ai')

        # Get confidences
        docai_conf = docai_result.get('confidence', 0.0)
        local_conf = local_result.get('confidence', 0.0)

        logger.debug(f"DocAI confidence: {docai_conf:.3f}, Local confidence: {local_conf:.3f}")

        # High confidence DocAI - use it
        if docai_conf >= self.docai_accept_threshold:
            logger.info("Using DocAI result (high confidence)")
            return self._select_result(docai_result, 'document_ai')

        # Low confidence DocAI - use local
        if docai_conf < self.docai_fallback_threshold:
            logger.info("Using local result (low DocAI confidence)")
            return self._select_result(local_result, 'local')

        # Medium confidence - ensemble
        if self.ensemble_range[0] <= docai_conf <= self.ensemble_range[1]:
            logger.info("Using ensemble (medium confidence)")
            return self._ensemble(docai_result, local_result)

        # Default to higher confidence
        if docai_conf >= local_conf:
            return self._select_result(docai_result, 'document_ai')
        else:
            return self._select_result(local_result, 'local')

    def _ensemble(self, docai_result: Dict, local_result: Dict) -> Dict:
        """
        Ensemble multiple results.

        Args:
            docai_result: DocAI result
            local_result: Local model result

        Returns:
            Ensembled result
        """
        if self.ensemble_method == 'max_confidence':
            return self._ensemble_max_confidence(docai_result, local_result)
        elif self.ensemble_method == 'weighted_average':
            return self._ensemble_weighted(docai_result, local_result)
        elif self.ensemble_method == 'voting':
            return self._ensemble_voting(docai_result, local_result)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _ensemble_max_confidence(self, docai_result: Dict, local_result: Dict) -> Dict:
        """Select result with maximum confidence."""
        docai_conf = docai_result.get('confidence', 0.0)
        local_conf = local_result.get('confidence', 0.0)

        if docai_conf >= local_conf:
            result = docai_result.copy()
            result['routing'] = 'ensemble_max_confidence_docai'
        else:
            result = local_result.copy()
            result['routing'] = 'ensemble_max_confidence_local'

        return result

    def _ensemble_weighted(self, docai_result: Dict, local_result: Dict) -> Dict:
        """Weighted average ensemble (simple text selection for now)."""
        # For text, we use weighted confidence to select
        docai_conf = docai_result.get('confidence', 0.0)
        local_conf = local_result.get('confidence', 0.0)

        # Weighted confidence
        weighted_docai = docai_conf * self.ensemble_weights[0]
        weighted_local = local_conf * self.ensemble_weights[1]

        if weighted_docai >= weighted_local:
            result = docai_result.copy()
            result['routing'] = 'ensemble_weighted_docai'
        else:
            result = local_result.copy()
            result['routing'] = 'ensemble_weighted_local'

        # Average confidence
        result['confidence'] = (weighted_docai + weighted_local) / sum(self.ensemble_weights)

        return result

    def _ensemble_voting(self, docai_result: Dict, local_result: Dict) -> Dict:
        """Voting ensemble (placeholder for character-level voting)."""
        # Simplified: just use max confidence for now
        # In production, this would do character-level or word-level voting
        return self._ensemble_max_confidence(docai_result, local_result)

    def _select_result(self, result: Dict, source: str) -> Dict:
        """
        Select a result and add routing metadata.

        Args:
            result: OCR result
            source: Result source

        Returns:
            Result with routing metadata
        """
        final_result = result.copy()
        final_result['routing'] = f'selected_{source}'
        return final_result

    def should_use_ensemble(self, confidence: float) -> bool:
        """Check if confidence is in ensemble range."""
        return self.ensemble_range[0] <= confidence <= self.ensemble_range[1]

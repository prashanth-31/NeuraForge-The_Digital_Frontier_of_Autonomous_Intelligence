from __future__ import annotations

from dataclasses import dataclass

from app.core.config import ScoringSettings
from app.services.tools import ToolInvocationResult


@dataclass(slots=True)
class ConfidenceBreakdown:
    base: float
    evidence: float
    tool_reliability: float
    self_assessment: float

    def as_dict(self) -> dict[str, float]:
        return {
            "base": self.base,
            "evidence": self.evidence,
            "tool_reliability": self.tool_reliability,
            "self_assessment": self.self_assessment,
        }


@dataclass(slots=True)
class ConfidenceResult:
    score: float
    breakdown: ConfidenceBreakdown

    def as_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            "score": self.score,
            "breakdown": self.breakdown.as_dict(),
        }


class ConfidenceScorer:
    """Blend multiple signals into a normalized confidence value."""

    def __init__(self, settings: ScoringSettings) -> None:
        self._settings = settings

    def score(
        self,
        *,
        evidence_count: int,
        tool_result: ToolInvocationResult | None,
        self_assessment: float | None,
    ) -> ConfidenceResult:
        evidence_ratio = min(max(evidence_count, 0) / self._settings.max_evidence, 1.0)
        tool_component = self._tool_reliability(tool_result)
        self_assessment_component = self._clamp(self_assessment if self_assessment is not None else 0.5)

        base = self._settings.base_confidence
        evidence_value = self._settings.evidence_weight * evidence_ratio
        tool_value = self._settings.tool_reliability_weight * tool_component
        self_assessment_value = self._settings.self_assessment_weight * self_assessment_component

        total = self._clamp(base + evidence_value + tool_value + self_assessment_value)
        breakdown = ConfidenceBreakdown(
            base=round(base, 4),
            evidence=round(evidence_value, 4),
            tool_reliability=round(tool_value, 4),
            self_assessment=round(self_assessment_value, 4),
        )
        return ConfidenceResult(score=round(total, 4), breakdown=breakdown)

    def _tool_reliability(self, tool_result: ToolInvocationResult | None) -> float:
        if tool_result is None:
            return 0.5

        cache_bonus = 0.2 if tool_result.cached else 0.0
        latency_score = self._latency_factor(tool_result.latency)
        blended = latency_score + cache_bonus
        return self._clamp(blended)

    @staticmethod
    def _latency_factor(latency: float | None) -> float:
        if latency is None:
            return 0.6
        # Latency <= 1s considered excellent, >= 6s progressively worse.
        normalized = 1.1 - (latency / 6.0)
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _clamp(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


@dataclass(slots=True)
class OutputQualityBreakdown:
    """Breakdown of output quality scoring components."""
    length_score: float       # Response length adequacy
    completeness_score: float # Addresses all aspects of the request
    coherence_score: float    # Logical flow and consistency
    specificity_score: float  # Contains concrete details vs vague generalizations
    tool_usage_score: float   # Appropriate use of tools (if applicable)
    
    def as_dict(self) -> dict[str, float]:
        return {
            "length_score": self.length_score,
            "completeness_score": self.completeness_score,
            "coherence_score": self.coherence_score,
            "specificity_score": self.specificity_score,
            "tool_usage_score": self.tool_usage_score,
        }


@dataclass(slots=True)
class OutputQualityResult:
    """Result of output quality assessment."""
    score: float
    breakdown: OutputQualityBreakdown
    issues: list[str]
    should_retry: bool
    
    def as_dict(self) -> dict[str, float | dict[str, float] | list[str] | bool]:
        return {
            "score": self.score,
            "breakdown": self.breakdown.as_dict(),
            "issues": self.issues,
            "should_retry": self.should_retry,
        }


class OutputQualityScorer:
    """
    Evaluates the quality of agent outputs to determine if they meet
    acceptable standards or should trigger a retry.
    
    Quality factors:
    - Length: Is the response appropriately detailed?
    - Completeness: Does it address all aspects of the request?
    - Coherence: Is it logically structured and consistent?
    - Specificity: Does it contain concrete details?
    - Tool usage: Were tools used appropriately (if expected)?
    """
    
    def __init__(
        self,
        *,
        min_length: int = 50,
        ideal_length: int = 200,
        retry_threshold: float = 0.4,
        length_weight: float = 0.15,
        completeness_weight: float = 0.30,
        coherence_weight: float = 0.25,
        specificity_weight: float = 0.20,
        tool_usage_weight: float = 0.10,
    ) -> None:
        self._min_length = min_length
        self._ideal_length = ideal_length
        self._retry_threshold = retry_threshold
        self._weights = {
            "length": length_weight,
            "completeness": completeness_weight,
            "coherence": coherence_weight,
            "specificity": specificity_weight,
            "tool_usage": tool_usage_weight,
        }
    
    def score(
        self,
        *,
        response: str,
        request: str,
        tools_used: list[str] | None = None,
        tools_expected: list[str] | None = None,
        confidence: float = 0.5,
    ) -> OutputQualityResult:
        """
        Score the quality of an agent output.
        
        Args:
            response: The agent's response text
            request: The original user request
            tools_used: List of tools that were actually used
            tools_expected: List of tools that were expected to be used
            confidence: The agent's self-reported confidence
            
        Returns:
            OutputQualityResult with score breakdown and retry recommendation
        """
        issues: list[str] = []
        
        # Score each component
        length_score = self._score_length(response, issues)
        completeness_score = self._score_completeness(response, request, issues)
        coherence_score = self._score_coherence(response, issues)
        specificity_score = self._score_specificity(response, issues)
        tool_score = self._score_tool_usage(tools_used, tools_expected, issues)
        
        # Compute weighted total
        total = (
            self._weights["length"] * length_score +
            self._weights["completeness"] * completeness_score +
            self._weights["coherence"] * coherence_score +
            self._weights["specificity"] * specificity_score +
            self._weights["tool_usage"] * tool_score
        )
        
        # Apply confidence modifier (low confidence → lower quality score)
        confidence_modifier = 0.8 + (0.2 * confidence)  # Range: 0.8 to 1.0
        final_score = self._clamp(total * confidence_modifier)
        
        breakdown = OutputQualityBreakdown(
            length_score=round(length_score, 4),
            completeness_score=round(completeness_score, 4),
            coherence_score=round(coherence_score, 4),
            specificity_score=round(specificity_score, 4),
            tool_usage_score=round(tool_score, 4),
        )
        
        should_retry = final_score < self._retry_threshold
        
        return OutputQualityResult(
            score=round(final_score, 4),
            breakdown=breakdown,
            issues=issues,
            should_retry=should_retry,
        )
    
    def _score_length(self, response: str, issues: list[str]) -> float:
        """Score response based on length adequacy."""
        if not response:
            issues.append("Empty response")
            return 0.0
        
        length = len(response)
        
        # Too short
        if length < self._min_length:
            issues.append(f"Response too short ({length} chars, min {self._min_length})")
            return length / self._min_length
        
        # Ideal range
        if length <= self._ideal_length * 2:
            return 1.0
        
        # Very long responses might indicate rambling
        if length > self._ideal_length * 5:
            issues.append("Response excessively long, may be unfocused")
            return 0.7
        
        return 0.9
    
    def _score_completeness(self, response: str, request: str, issues: list[str]) -> float:
        """Score how well the response addresses the request."""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        request_lower = request.lower()
        
        # Check for PASS responses (intentional non-answers)
        if "[pass]" in response_lower or response.strip() == "[PASS]":
            return 1.0  # PASS is a valid response
        
        # Extract key terms from request
        key_terms = self._extract_key_terms(request_lower)
        
        if not key_terms:
            return 0.8  # Can't evaluate without key terms
        
        # Count how many key terms are addressed
        addressed = sum(1 for term in key_terms if term in response_lower)
        coverage = addressed / len(key_terms)
        
        if coverage < 0.3:
            issues.append(f"Low topic coverage ({addressed}/{len(key_terms)} key terms)")
        
        return self._clamp(coverage + 0.2)  # Base score of 0.2
    
    def _score_coherence(self, response: str, issues: list[str]) -> float:
        """Score logical flow and consistency."""
        if not response:
            return 0.0
        
        # Check for structural elements that indicate organization
        score = 0.6  # Base score
        
        # Bonus for paragraphs (newlines)
        if "\n\n" in response:
            score += 0.1
        
        # Bonus for lists/bullet points
        if any(marker in response for marker in ["- ", "* ", "1. ", "• "]):
            score += 0.1
        
        # Check for transitional words indicating flow
        transitions = ["however", "therefore", "additionally", "furthermore", "in conclusion", "first", "second", "finally"]
        if any(t in response.lower() for t in transitions):
            score += 0.1
        
        # Penalty for repetition (same sentence appearing twice)
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        if len(sentences) > len(set(sentences)):
            score -= 0.2
            issues.append("Detected repetitive content")
        
        return self._clamp(score)
    
    def _score_specificity(self, response: str, issues: list[str]) -> float:
        """Score presence of concrete details vs vague generalizations."""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        
        # Check for PASS responses
        if "[pass]" in response_lower or response.strip() == "[PASS]":
            return 1.0  # PASS is appropriately specific
        
        score = 0.5  # Base score
        
        # Check for numbers/statistics (indicates specificity)
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            score += min(0.2, 0.05 * len(numbers))
        
        # Check for concrete terms
        concrete_indicators = [
            "specifically", "exactly", "for example", "such as", "including",
            "approximately", "about", "around", "between", "million", "billion",
            "percent", "%", "$", "€", "£", "¥",
        ]
        if any(ind in response_lower for ind in concrete_indicators):
            score += 0.2
        
        # Penalty for vague language
        vague_terms = [
            "might be", "could be", "possibly", "perhaps", "maybe",
            "something like", "some kind of", "in general",
        ]
        vague_count = sum(1 for term in vague_terms if term in response_lower)
        if vague_count > 3:
            score -= 0.15
            issues.append("Response contains excessive vague language")
        
        return self._clamp(score)
    
    def _score_tool_usage(
        self,
        tools_used: list[str] | None,
        tools_expected: list[str] | None,
        issues: list[str],
    ) -> float:
        """Score appropriate use of tools."""
        # If no tools expected, full score
        if not tools_expected:
            return 1.0
        
        tools_used = tools_used or []
        
        # Check overlap
        used_set = set(tools_used)
        expected_set = set(tools_expected)
        
        if not expected_set:
            return 1.0
        
        overlap = used_set & expected_set
        coverage = len(overlap) / len(expected_set)
        
        if coverage < 0.5:
            issues.append(f"Expected tools not used: {expected_set - used_set}")
        
        return self._clamp(coverage)
    
    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from a request for completeness checking."""
        # Remove common stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "to", "of",
            "in", "on", "at", "for", "with", "about", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "also", "now", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "it", "its", "i", "me", "my", "mine",
            "we", "our", "ours", "us", "you", "your", "yours", "he", "she",
            "him", "her", "his", "hers", "they", "them", "their", "theirs",
            "if", "and", "but", "or", "because", "as", "until", "while",
        }
        
        # Simple tokenization
        import re
        words = re.findall(r'\b[a-z]+\b', text)
        
        # Filter and return significant terms
        terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Return unique terms, limit to prevent over-fitting
        return list(dict.fromkeys(terms))[:10]
    
    @staticmethod
    def _clamp(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
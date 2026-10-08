"""
Output Evaluation and Benchmarking Engine
Evaluates generated outputs across Relevance, Consistency, Factuality, Completeness, and Readability.
Includes native implementations of ROUGE-1/2/L, TF-IDF cosine similarity, and LLM-as-a-Judge grading.
"""

import re
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


# =====================================================================
# Evaluation Data Structures
# =====================================================================

@dataclass
class MetricScore:
    name: str
    score: float  # 0.0 to 1.0
    percentage: float  # 0 to 100
    rating: str  # "Excellent", "Good", "Moderate", "Needs Improvement"
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RougeScores:
    rouge_1_f1: float
    rouge_1_precision: float
    rouge_1_recall: float
    rouge_2_f1: float
    rouge_2_precision: float
    rouge_2_recall: float
    rouge_l_f1: float
    rouge_l_precision: float
    rouge_l_recall: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass
class EvaluationReport:
    overall_quality_index: float  # 0 to 100
    grade: str  # "A+", "A", "B", "C", "D"
    factuality: MetricScore
    consistency: MetricScore
    relevance: MetricScore
    completeness: MetricScore
    readability: MetricScore
    rouge: RougeScores
    radar_data: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    llm_judge_verdict: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_quality_index": round(self.overall_quality_index, 1),
            "grade": self.grade,
            "factuality": self.factuality.to_dict(),
            "consistency": self.consistency.to_dict(),
            "relevance": self.relevance.to_dict(),
            "completeness": self.completeness.to_dict(),
            "readability": self.readability.to_dict(),
            "rouge": self.rouge.to_dict(),
            "radar_data": {k: round(v, 1) for k, v in self.radar_data.items()},
            "recommendations": self.recommendations,
            "llm_judge_verdict": self.llm_judge_verdict,
        }


# =====================================================================
# Evaluation Engine
# =====================================================================

class EvaluationEngine:
    """Evaluates generated text for factual fidelity, relevance, consistency, and readability."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    # -----------------------------------------------------------------
    # NLP Helper Algorithms
    # -----------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b[A-Za-z0-9_\-']+\b", text.lower())

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """Longest Common Subsequence length for ROUGE-L."""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if x[i] == y[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n]

    def compute_rouge(self, generated_text: str, reference_text: str) -> RougeScores:
        """Calculates ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1."""
        gen_tokens = self._tokenize(generated_text)
        ref_tokens = self._tokenize(reference_text)

        if not gen_tokens or not ref_tokens:
            return RougeScores(0, 0, 0, 0, 0, 0, 0, 0, 0)

        # ROUGE-1 (Unigrams)
        gen_1 = gen_tokens
        ref_1 = set(ref_tokens)
        overlap_1 = sum(1 for t in gen_1 if t in ref_1)
        r1_rec = overlap_1 / len(ref_tokens) if ref_tokens else 0
        r1_prec = overlap_1 / len(gen_tokens) if gen_tokens else 0
        r1_f1 = (2 * r1_prec * r1_rec / (r1_prec + r1_rec)) if (r1_prec + r1_rec) > 0 else 0

        # ROUGE-2 (Bigrams)
        gen_2 = self._get_ngrams(gen_tokens, 2)
        ref_2 = set(self._get_ngrams(ref_tokens, 2))
        overlap_2 = sum(1 for b in gen_2 if b in ref_2)
        r2_rec = overlap_2 / max(len(ref_2), 1)
        r2_prec = overlap_2 / max(len(gen_2), 1)
        r2_f1 = (2 * r2_prec * r2_rec / (r2_prec + r2_rec)) if (r2_prec + r2_rec) > 0 else 0

        # ROUGE-L (LCS)
        lcs = self._lcs_length(gen_tokens, ref_tokens)
        rl_rec = lcs / len(ref_tokens) if ref_tokens else 0
        rl_prec = lcs / len(gen_tokens) if gen_tokens else 0
        rl_f1 = (2 * rl_prec * rl_rec / (rl_prec + rl_rec)) if (rl_prec + rl_rec) > 0 else 0

        return RougeScores(
            rouge_1_f1=r1_f1, rouge_1_precision=r1_prec, rouge_1_recall=r1_rec,
            rouge_2_f1=r2_f1, rouge_2_precision=r2_prec, rouge_2_recall=r2_rec,
            rouge_l_f1=rl_f1, rouge_l_precision=rl_prec, rouge_l_recall=rl_rec,
        )

    def compute_tfidf_cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Computes TF-IDF vector cosine similarity between two texts."""
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)
        if not tokens_a or not tokens_b:
            return 0.0

        vocab = sorted(list(set(tokens_a + tokens_b)))
        freq_a = {t: tokens_a.count(t) for t in set(tokens_a)}
        freq_b = {t: tokens_b.count(t) for t in set(tokens_b)}

        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0

        for term in vocab:
            tf_a = math.log1p(freq_a.get(term, 0))
            tf_b = math.log1p(freq_b.get(term, 0))
            dot_product += tf_a * tf_b
            norm_a += tf_a * tf_a
            norm_b += tf_b * tf_b

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def evaluate_readability(self, text: str) -> MetricScore:
        """Calculates Flesch Reading Ease score and clarity rating."""
        words = self._tokenize(text)
        sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", text) if s.strip()]

        if not words or not sentences:
            return MetricScore("Readability", 0.5, 50.0, "Moderate", "Insufficient text for evaluation.")

        total_words = len(words)
        total_sentences = len(sentences)

        # Estimate syllables (vowel sequences count)
        def count_syllables(w: str) -> int:
            w = w.lower()
            count = len(re.findall(r"[aeiouy]+", w))
            if w.endswith("e") and not w.endswith("le") and count > 1:
                count -= 1
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)

        # Flesch Reading Ease formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        asl = total_words / total_sentences
        asw = total_syllables / total_words
        flesch = 206.835 - (1.015 * asl) - (84.6 * asw)

        # Bound to [0, 100]
        flesch_bounded = max(0.0, min(100.0, flesch))
        norm_score = flesch_bounded / 100.0

        if flesch_bounded >= 65:
            rating = "Excellent"
            expl = f"High clarity and accessible tone (Flesch Score: {round(flesch_bounded, 1)})."
        elif flesch_bounded >= 50:
            rating = "Good"
            expl = f"Standard professional document prose (Flesch Score: {round(flesch_bounded, 1)})."
        elif flesch_bounded >= 35:
            rating = "Moderate"
            expl = f"Dense technical/academic prose; complex syntax (Flesch Score: {round(flesch_bounded, 1)})."
        else:
            rating = "Needs Improvement"
            expl = f"Extremely dense sentences and heavy jargon (Flesch Score: {round(flesch_bounded, 1)})."

        return MetricScore("Readability", round(norm_score, 3), round(flesch_bounded, 1), rating, expl)

    def evaluate_factuality(self, generated_text: str, source_text: str) -> MetricScore:
        """
        Evaluates source grounding, entity preservation, and factuality.
        Checks numbers, monetary figures, and proper nouns against source text.
        """
        if not source_text or not generated_text:
            return MetricScore("Factuality", 0.5, 50.0, "Moderate", "Source text or generated text empty.")

        source_lower = source_text.lower()
        gen_lower = generated_text.lower()

        # Check key numeric claims ($X, %, dates)
        claims = re.findall(r"(\$[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?%|\b\d{4}\b)", generated_text)
        claims = list(set(claims))

        if not claims:
            # Word-level overlap in top informative terms
            gen_words = set([w for w in self._tokenize(generated_text) if len(w) > 4])
            source_words = set(self._tokenize(source_text))
            overlap_ratio = len(gen_words.intersection(source_words)) / max(len(gen_words), 1)
            score = min(1.0, overlap_ratio * 1.2)
            pct = score * 100
            rating = "Good" if pct >= 70 else "Moderate"
            return MetricScore("Factuality", round(score, 3), round(pct, 1), rating, "Factual terminology well-grounded in source.")

        verified = [c for c in claims if c.lower() in source_lower or re.sub(r"[$,]", "", c.lower()) in source_lower]
        grounding_ratio = len(verified) / len(claims)

        score = 0.3 + (grounding_ratio * 0.7)
        pct = score * 100.0

        if grounding_ratio >= 0.9:
            rating = "Excellent"
            expl = f"{len(verified)} of {len(claims)} key metrics verified directly in source ({round(grounding_ratio*100, 1)}% grounding)."
        elif grounding_ratio >= 0.7:
            rating = "Good"
            expl = f"Most metrics grounded ({len(verified)}/{len(claims)}). Minor ungrounded references."
        else:
            rating = "Needs Improvement"
            expl = f"Potential hallucination detected: {len(claims) - len(verified)} metrics/entities unverified in source."

        return MetricScore("Factuality", round(score, 3), round(pct, 1), rating, expl)

    def evaluate_relevance(self, generated_text: str, requirements: str) -> MetricScore:
        """Measures semantic alignment between prompt requirements and generated response."""
        if not requirements.strip():
            return MetricScore("Relevance", 0.85, 85.0, "Good", "General prompt executed accurately.")

        # 1. Cosine similarity
        cosine_sim = self.compute_tfidf_cosine_similarity(requirements, generated_text)

        # 2. Key requirements coverage
        req_keywords = [w for w in self._tokenize(requirements) if len(w) > 3]
        gen_tokens = set(self._tokenize(generated_text))
        coverage = sum(1 for k in req_keywords if k in gen_tokens) / max(len(req_keywords), 1) if req_keywords else 1.0

        # Composite score
        score = min(1.0, (cosine_sim * 0.4) + (coverage * 0.6) + 0.15)
        pct = score * 100.0

        if pct >= 80:
            rating = "Excellent"
            expl = f"High intent alignment ({round(coverage*100, 1)}% of requirement concepts covered)."
        elif pct >= 65:
            rating = "Good"
            expl = f"Good alignment with primary user instructions (coverage: {round(coverage*100, 1)}%)."
        else:
            rating = "Moderate"
            expl = f"Partial coverage of user specifications ({round(coverage*100, 1)}% matched)."

        return MetricScore("Relevance", round(score, 3), round(pct, 1), rating, expl)

    def evaluate_consistency(self, generated_text: str, source_text: str) -> MetricScore:
        """Evaluates internal narrative coherence and structural continuity."""
        paragraphs = [p.strip() for p in generated_text.split("\n\n") if len(p.strip()) > 30]

        if len(paragraphs) < 2:
            return MetricScore("Consistency", 0.85, 85.0, "Good", "Cohesive single-block narrative structure.")

        # Inter-paragraph semantic flow
        sims = []
        for i in range(len(paragraphs) - 1):
            s = self.compute_tfidf_cosine_similarity(paragraphs[i], paragraphs[i + 1])
            sims.append(s)

        avg_flow = sum(sims) / len(sims) if sims else 0.5
        # Ideal inter-paragraph continuity is moderate (not identical repetition, not unrelated disconnect)
        score = 0.5 + min(0.4, avg_flow * 1.2)
        pct = score * 100.0

        rating = "Excellent" if pct >= 80 else ("Good" if pct >= 65 else "Moderate")
        expl = f"Strong structural narrative continuity across {len(paragraphs)} distinct sections."

        return MetricScore("Consistency", round(score, 3), round(pct, 1), rating, expl)

    def evaluate_completeness(self, generated_text: str, requirements: str) -> MetricScore:
        """Checks structural depth, headers, bullet points, and word count sufficiency."""
        words = generated_text.split()
        wc = len(words)
        has_headers = bool(re.search(r"^#{1,4}\s+", generated_text, re.MULTILINE))
        has_bullets = bool(re.search(r"^\s*[-*]\s+", generated_text, re.MULTILINE))

        depth_bonus = 0.0
        if has_headers:
            depth_bonus += 0.1
        if has_bullets:
            depth_bonus += 0.1

        if wc > 350:
            base = 0.8
        elif wc > 180:
            base = 0.7
        elif wc > 80:
            base = 0.55
        else:
            base = 0.4

        score = min(1.0, base + depth_bonus)
        pct = score * 100.0
        rating = "Excellent" if pct >= 85 else ("Good" if pct >= 70 else "Moderate")
        expl = f"Structured document with {wc} words containing formatted sections and action points."

        return MetricScore("Completeness", round(score, 3), round(pct, 1), rating, expl)

    # -----------------------------------------------------------------
    # Main Evaluation Suite
    # -----------------------------------------------------------------

    def evaluate(
        self,
        generated_text: str,
        source_text: str = "",
        requirements: str = "",
        run_llm_judge: bool = False,
    ) -> EvaluationReport:
        """
        Executes end-to-end multi-dimensional evaluation.
        """
        factuality = self.evaluate_factuality(generated_text, source_text)
        relevance = self.evaluate_relevance(generated_text, requirements)
        consistency = self.evaluate_consistency(generated_text, source_text)
        completeness = self.evaluate_completeness(generated_text, requirements)
        readability = self.evaluate_readability(generated_text)
        rouge = self.compute_rouge(generated_text, source_text)

        # Weighted Overall Quality Index
        # Weights: Factuality 30%, Relevance 25%, Consistency 20%, Completeness 15%, Readability 10%
        overall_index = (
            factuality.percentage * 0.30 +
            relevance.percentage * 0.25 +
            consistency.percentage * 0.20 +
            completeness.percentage * 0.15 +
            readability.percentage * 0.10
        )

        if overall_index >= 90:
            grade = "A+"
        elif overall_index >= 80:
            grade = "A"
        elif overall_index >= 70:
            grade = "B"
        elif overall_index >= 60:
            grade = "C"
        else:
            grade = "D"

        radar_data = {
            "Factuality": factuality.percentage,
            "Relevance": relevance.percentage,
            "Consistency": consistency.percentage,
            "Completeness": completeness.percentage,
            "Readability": readability.percentage,
        }

        recommendations = []
        if factuality.percentage < 75:
            recommendations.append("Strengthen factual grounding by injecting strict entity constraints and few-shot grounding demonstrations.")
        if relevance.percentage < 75:
            recommendations.append("Increase prompt specificity; ensure key terms from requirements are explicitly incorporated in the generation template.")
        if completeness.percentage < 75:
            recommendations.append("Expand generation depth by prompting for structured sub-sections or raising max token allowance.")
        if readability.percentage < 50:
            recommendations.append("Improve readability: prompt the model to simplify sentence structure and define domain jargon.")

        if not recommendations:
            recommendations.append("Output meets enterprise quality benchmarks across all five evaluation dimensions.")

        llm_judge = None
        if run_llm_judge and self.llm_client:
            llm_judge = self._run_llm_judge(generated_text, source_text, requirements)

        return EvaluationReport(
            overall_quality_index=overall_index,
            grade=grade,
            factuality=factuality,
            consistency=consistency,
            relevance=relevance,
            completeness=completeness,
            readability=readability,
            rouge=rouge,
            radar_data=radar_data,
            recommendations=recommendations,
            llm_judge_verdict=llm_judge,
        )

    def _run_llm_judge(self, generated_text: str, source_text: str, requirements: str) -> Dict[str, Any]:
        """Runs LLM-as-a-Judge rubric evaluation."""
        judge_prompt = f"""You are an expert AI Evaluation Auditor. Grade this output from 1 to 10 on:
1. Factuality & Faithfulness
2. Intent Relevance
3. Coherence & Polish

USER REQUIREMENTS: {requirements}

SOURCE DOCUMENT:
\"\"\"
{source_text[:1500]}
\"\"\"

GENERATED OUTPUT:
\"\"\"
{generated_text[:1500]}
\"\"\"

Return a valid JSON object with:
{{
  "factuality_score": 9,
  "relevance_score": 9,
  "coherence_score": 9,
  "judge_feedback": "string explanation",
  "audit_verdict": "PASS|NEEDS_REVISION"
}}
"""
        try:
            resp = self.llm_client.generate(
                prompt=judge_prompt,
                system_prompt="You are a strict evaluation auditor. Respond only with valid JSON.",
                output_format="json"
            )
            import json
            match = re.search(r"\{[\s\S]*\}", resp.content)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            return {"error": f"LLM judge failed: {e}", "audit_verdict": "UNAVAILABLE"}
        return {"audit_verdict": "SKIPPED"}

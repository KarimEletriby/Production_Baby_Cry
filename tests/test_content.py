"""
Tests for api.content — text, logic, and labels formatting.
"""
from api.content import (
    STAGE1_CONTENT,
    STAGE2_CONTENT,
    confidence_label_ar,
    confidence_advice_ar,
)


class TestContentDictionaries:
    def test_stage1_content_has_all_keys(self):
        for label in ["baby_cry", "not_baby_cry"]:
            assert label in STAGE1_CONTENT
            content = STAGE1_CONTENT[label]
            assert "emoji" in content
            assert "title_ar" in content
            assert "description_ar" in content

    def test_stage2_content_has_all_keys(self):
        for label in ["scared", "physical_pain", "needs", "burping"]:
            assert label in STAGE2_CONTENT
            content = STAGE2_CONTENT[label]
            assert "emoji" in content
            assert "name_ar" in content
            assert "definition_ar" in content
            assert "physical_signs_ar" in content
            assert "tips_ar" in content
            assert "warning_ar" in content
            assert isinstance(content["physical_signs_ar"], list)
            assert isinstance(content["tips_ar"], list)


class TestConfidenceLabel:
    def test_high_confidence(self):
        assert confidence_label_ar(0.95) == "ثقة عالية جداً"
        assert confidence_label_ar(0.85) == "ثقة عالية جداً"

    def test_medium_high_confidence(self):
        assert confidence_label_ar(0.75) == "ثقة عالية"
        assert confidence_label_ar(0.70) == "ثقة عالية"

    def test_medium_confidence(self):
        assert confidence_label_ar(0.60) == "ثقة متوسطة"
        assert confidence_label_ar(0.55) == "ثقة متوسطة"

    def test_low_confidence(self):
        assert confidence_label_ar(0.45) == "ثقة منخفضة"
        assert confidence_label_ar(0.40) == "ثقة منخفضة"

    def test_very_low_confidence(self):
        assert confidence_label_ar(0.35) == "غير متأكد"


class TestConfidenceAdvice:
    def test_burping_always_warns(self):
        advice = confidence_advice_ar(0.99, cry_type="burping")
        assert "أقل دقة" in advice
        assert "ملاحظتك لطفلك" in advice

    def test_low_confidence_general_warning(self):
        advice = confidence_advice_ar(0.50, cry_type="needs")
        assert "غير متأكد تماماً" in advice
        assert "أقل دقة" not in advice  # Burping warning shouldn't be here

    def test_medium_confidence_general_warning(self):
        advice = confidence_advice_ar(0.65, cry_type="scared")
        assert "مستوى الثقة متوسط" in advice

    def test_high_confidence_no_warning(self):
        advice = confidence_advice_ar(0.85, cry_type="physical_pain")
        assert advice == ""

    def test_low_confidence_burping_combines_warnings(self):
        advice = confidence_advice_ar(0.50, cry_type="burping")
        assert "أقل دقة" in advice
        assert "غير متأكد تماماً" in advice
        assert "\n" in advice

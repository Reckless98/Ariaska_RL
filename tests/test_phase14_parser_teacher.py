#!/usr/bin/env python3
"""
tests/test_phase14_parser_teacher.py — Phase 14.0: ParserTeacherOutput Tests

Contract C3.5: 5 required tests for parser teacher output, learning features, parsing lessons.
"""

import os
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestParserTeacherOutput:
    """C3.5: ParserTeacherOutput schema and feature vector tests."""

    def test_empty_output(self):
        """ParserTeacherOutput.empty() creates valid empty output."""
        from core.execution.parser_teacher_output import ParserTeacherOutput
        out = ParserTeacherOutput.empty()
        assert out.has_discoveries() is False
        assert len(out.events) == 0
        assert len(out.lessons) == 0
        assert len(out.confidence_disagreements) == 0

    def test_learning_features_vector_shape(self):
        """LearningFeatures.feature_vector() returns exactly (32,)."""
        import torch
        from core.execution.parser_teacher_output import LearningFeatures
        lf = LearningFeatures(
            discovery_counts={"open_port": 3, "service": 2},
            confidence_mean=0.85,
            confidence_min=0.6,
            stage_reached=2,
            novel_discoveries=4,
            repeated_discoveries=1,
            total_events=5,
        )
        vec = lf.feature_vector(max_dims=32)
        assert isinstance(vec, torch.Tensor)
        assert vec.shape == (32,)

    def test_parsing_lesson_bounded(self):
        """ParsingLesson example_output truncated to ≤128 chars."""
        from core.execution.parser_teacher_output import ParsingLesson
        lesson = ParsingLesson(
            pattern=r"\d+/tcp\s+open",
            discovery_type="open_port",
            confidence=0.9,
            example_output="x" * 300,
        )
        assert len(lesson.example_output) <= 128

    def test_max_lessons_enforced(self):
        """ParserTeacherOutput enforces max 3 lessons."""
        from core.execution.parser_teacher_output import ParserTeacherOutput, ParsingLesson
        lessons = [
            ParsingLesson(pattern=f"p{i}", discovery_type="open_port", confidence=0.5)
            for i in range(10)
        ]
        out = ParserTeacherOutput(lessons=lessons)
        assert len(out.lessons) <= 3

    def test_max_confidence_disagreements_enforced(self):
        """ParserTeacherOutput enforces max 3 confidence_disagreements."""
        from core.execution.parser_teacher_output import ParserTeacherOutput
        disagreements = [
            {"stage1": 0.5, "stage2": 0.9} for _ in range(10)
        ]
        out = ParserTeacherOutput(confidence_disagreements=disagreements)
        assert len(out.confidence_disagreements) <= 3

"""Tests for audio chunking utilities."""

import numpy as np
import pytest
import torch

from sam_audio_infer.chunking import AudioChunker, ChunkingConfig


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self):
        config = ChunkingConfig()
        assert config.chunk_duration == 25.0
        assert config.sample_rate == 16000
        assert config.overlap_duration == 0.0

    def test_chunk_samples(self):
        config = ChunkingConfig(chunk_duration=10.0, sample_rate=16000)
        assert config.chunk_samples == 160000

    def test_invalid_chunk_duration(self):
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_duration=-1.0)

    def test_overlap_exceeds_chunk(self):
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_duration=10.0, overlap_duration=15.0)


class TestAudioChunker:
    """Tests for AudioChunker."""

    def test_needs_chunking_short_audio(self):
        chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
        # 5 seconds of audio
        audio = torch.randn(80000)
        assert not chunker.needs_chunking(audio)

    def test_needs_chunking_long_audio(self):
        chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
        # 30 seconds of audio
        audio = torch.randn(480000)
        assert chunker.needs_chunking(audio)

    def test_get_num_chunks(self):
        chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
        # 25 seconds of audio
        audio = torch.randn(400000)
        assert chunker.get_num_chunks(audio) == 3

    def test_chunk_iteration(self):
        chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
        audio = torch.randn(400000)  # 25 seconds

        chunks = list(chunker.chunk(audio))
        assert len(chunks) == 3
        assert chunks[0].is_first
        assert chunks[-1].is_last
        assert chunks[0].chunk_index == 0
        assert chunks[-1].chunk_index == 2

    def test_merge_chunks(self):
        chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)

        # Create some fake chunks
        chunks = [torch.randn(160000) for _ in range(3)]
        merged = chunker.merge(chunks, crossfade=False)

        # Should be concatenated
        assert merged.shape[0] == 480000

    def test_chunk_with_stereo(self):
        chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
        audio = torch.randn(400000, 2)  # 25 seconds, stereo

        chunks = list(chunker.chunk(audio))
        assert len(chunks) == 3
        assert chunks[0].data.shape[1] == 2  # Stereo preserved

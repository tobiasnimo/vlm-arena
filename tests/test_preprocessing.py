import pytest

from utils.preprocessing import make_chunks, parse_frame_timestamp


# ── parse_frame_timestamp ─────────────────────────────────────────────────────

class TestParseFrameTimestamp:
    def test_basic(self):
        assert parse_frame_timestamp("frame_00-00-05-000.jpg") == "00:00:05"

    def test_minutes(self):
        assert parse_frame_timestamp("frame_00-01-30-000.jpg") == "00:01:30"

    def test_hours(self):
        assert parse_frame_timestamp("frame_01-02-03-500.jpg") == "01:02:03"

    def test_full_path(self, tmp_path):
        p = tmp_path / "frame_00-00-42-123.jpg"
        p.touch()
        assert parse_frame_timestamp(str(p)) == "00:00:42"

    def test_milliseconds_ignored(self):
        """Only HH:MM:SS is returned; milliseconds are discarded."""
        assert parse_frame_timestamp("frame_00-00-10-999.jpg") == "00:00:10"

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            parse_frame_timestamp("not_a_frame.jpg")

    def test_wrong_format_raises(self):
        with pytest.raises(ValueError):
            parse_frame_timestamp("frame_0-0-0-0.jpg")


# ── make_chunks ───────────────────────────────────────────────────────────────

class TestMakeChunks:
    def _frames(self, n):
        return [f"f{i}" for i in range(n)]

    def test_no_overlap_exact_fit(self):
        chunks = make_chunks(self._frames(10), chunk_size=5, overlap=0)
        assert chunks == [["f0","f1","f2","f3","f4"], ["f5","f6","f7","f8","f9"]]

    def test_no_overlap_with_remainder(self):
        chunks = make_chunks(self._frames(7), chunk_size=5, overlap=0)
        assert len(chunks) == 2
        assert chunks[0] == ["f0","f1","f2","f3","f4"]
        assert chunks[1] == ["f5","f6"]

    def test_with_overlap(self):
        chunks = make_chunks(self._frames(10), chunk_size=5, overlap=2)
        # step = 3 → starts at 0, 3, 6, 9
        assert chunks[0] == ["f0","f1","f2","f3","f4"]
        assert chunks[1] == ["f3","f4","f5","f6","f7"]
        assert chunks[2] == ["f6","f7","f8","f9"]

    def test_overlap_one(self):
        chunks = make_chunks(self._frames(6), chunk_size=3, overlap=1)
        # step = 2 → starts at 0, 2, 4
        assert chunks[0] == ["f0","f1","f2"]
        assert chunks[1] == ["f2","f3","f4"]
        assert chunks[2] == ["f4","f5"]

    def test_single_chunk(self):
        chunks = make_chunks(self._frames(3), chunk_size=5, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == ["f0","f1","f2"]

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            make_chunks(self._frames(10), chunk_size=5, overlap=5)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError):
            make_chunks(self._frames(10), chunk_size=5, overlap=6)

    def test_empty_input(self):
        assert make_chunks([], chunk_size=5, overlap=0) == []

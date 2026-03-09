import PIL.Image
import pytest

from utils.inference import MockVLMModel, load_images, load_model


# ── MockVLMModel ──────────────────────────────────────────────────────────────

class TestMockVLMModel:
    def test_run_returns_string_and_float(self):
        model = MockVLMModel(key="phi3_v", label="Phi-3.5-Vision")
        answer, elapsed = model.run("What do you see?", ["img1", "img2"])
        assert isinstance(answer, str)
        assert isinstance(elapsed, float)

    def test_answer_contains_mock_marker(self):
        model = MockVLMModel(key="phi3_v", label="Phi-3.5-Vision")
        answer, _ = model.run("Q", ["img"])
        assert "[MOCK" in answer

    def test_answer_reflects_frame_count(self):
        model = MockVLMModel(key="qwen2_vl", label="Qwen2-VL-7B")
        answer, _ = model.run("Q", ["a", "b", "c"])
        assert "3" in answer

    def test_elapsed_is_fixed(self):
        model = MockVLMModel(key="phi3_v", label="Phi-3.5-Vision")
        _, elapsed = model.run("Q", ["img"])
        assert elapsed == 0.05

    def test_key_and_label_stored(self):
        model = MockVLMModel(key="deepseek_vl2", label="DeepSeek-VL2-Tiny")
        assert model.key == "deepseek_vl2"
        assert model.label == "DeepSeek-VL2-Tiny"


# ── load_model factory ────────────────────────────────────────────────────────

class TestLoadModel:
    def test_mock_returns_mock_model(self):
        model = load_model("phi3_v", chunk_size=5, mock=True)
        assert isinstance(model, MockVLMModel)
        assert model.key == "phi3_v"

    def test_mock_all_registered_keys(self):
        keys = ["phi3_v", "qwen2_vl", "llava_next", "internvl2", "deepseek_vl2"]
        for key in keys:
            model = load_model(key, chunk_size=5, mock=True)
            assert model.key == key

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown model key"):
            load_model("nonexistent_model", chunk_size=5, mock=True)

    def test_real_load_not_called_when_mock(self, mocker):
        """Verify no GPU loader is invoked when mock=True."""
        spy = mocker.patch("utils.inference.load_phi3_v")
        load_model("phi3_v", chunk_size=5, mock=True)
        spy.assert_not_called()


# ── load_images ───────────────────────────────────────────────────────────────

class TestLoadImages:
    def test_loads_local_jpeg(self, tmp_path):
        img = PIL.Image.new("RGB", (16, 16), color=(255, 0, 0))
        path = tmp_path / "test.jpg"
        img.save(path, format="JPEG")

        result = load_images([str(path)])
        assert len(result) == 1
        assert isinstance(result[0], PIL.Image.Image)
        assert result[0].mode == "RGB"

    def test_loads_multiple_images(self, tmp_path):
        for i in range(3):
            img = PIL.Image.new("RGB", (8, 8), color=(i * 80, 0, 0))
            img.save(tmp_path / f"img_{i}.jpg", format="JPEG")

        paths = [str(tmp_path / f"img_{i}.jpg") for i in range(3)]
        result = load_images(paths)
        assert len(result) == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Image not found"):
            load_images(["/nonexistent/path/image.jpg"])

    def test_non_rgb_converted_to_rgb(self, tmp_path):
        img = PIL.Image.new("RGBA", (8, 8), color=(0, 255, 0, 128))
        path = tmp_path / "rgba.png"
        img.save(path, format="PNG")

        result = load_images([str(path)])
        assert result[0].mode == "RGB"

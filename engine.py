"""
Engine bọc quanh package `qwen-tts` của QwenLM.

Nhiệm vụ:
  - Nạp / đổi / giải phóng model 1.7B trên GPU (chỉ giữ 1 model trong VRAM).
  - Cắt văn bản dài thành từng đoạn rồi ghép audio lại.
  - Gói 3 chế độ sinh giọng vào chung một hàm `synthesize`.

Tài liệu gốc: https://github.com/QwenLM/Qwen3-TTS
"""

from __future__ import annotations

import gc
import os
import re
import sys
import threading
from typing import Callable, Iterable

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Hằng số
# --------------------------------------------------------------------------- #

# 3 biến thể 1.7B. Mỗi biến thể là 1 checkpoint riêng, không dùng chung được.
MODEL_IDS = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# Bản 0.6B cho máy VRAM thấp. Không có VoiceDesign nên chế độ đó luôn rơi về 1.7B.
MODEL_IDS_06B = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "voice_clone": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

MODE_LABELS = {
    "custom_voice": "Giọng có sẵn",
    "voice_design": "Thiết kế giọng",
    "voice_clone": "Nhân bản giọng",
}

# "Auto" = để model tự nhận diện ngôn ngữ.
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]

# 9 giọng dựng sẵn của model CustomVoice.
SPEAKERS = {
    "Vivian": "Nữ trẻ, sáng, hơi gằn — tiếng Trung",
    "Serena": "Nữ trẻ, ấm, dịu — tiếng Trung",
    "Uncle_Fu": "Nam trung niên, trầm, mượt — tiếng Trung",
    "Dylan": "Nam trẻ giọng Bắc Kinh, trong, tự nhiên",
    "Eric": "Nam giọng Thành Đô, hơi khàn, linh hoạt",
    "Ryan": "Nam, tiết tấu mạnh, dứt khoát — tiếng Anh",
    "Aiden": "Nam Mỹ, tươi sáng, trung âm rõ — tiếng Anh",
    "Ono_Anna": "Nữ Nhật, nhẹ, nhanh, tinh nghịch",
    "Sohee": "Nữ Hàn, ấm, giàu cảm xúc",
}

SILENCE_BETWEEN_CHUNKS = 0.18  # giây chèn giữa các đoạn khi ghép
DEFAULT_SAMPLE_RATE = 24000    # chỉ dùng khi không đọc được sr từ model


def log(msg: str) -> None:
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


def is_oom(exc: BaseException) -> bool:
    """Phân biệt lỗi hết VRAM với lỗi lập trình, để không nuốt nhầm."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return "out of memory" in str(exc).lower()


class OutOfVRAM(RuntimeError):
    """Lỗi hết VRAM, kèm gợi ý xử lý bằng tiếng Việt."""


# --------------------------------------------------------------------------- #
# Cắt văn bản
# --------------------------------------------------------------------------- #

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？；;…\n])\s*")
_CLAUSE_SPLIT = re.compile(r"(?<=[,，、:：])\s*")


def _hard_wrap(piece: str, limit: int) -> list[str]:
    """Cắt cứng một câu quá dài: ưu tiên dấu phẩy, cuối cùng mới cắt theo từ."""
    if len(piece) <= limit:
        return [piece]

    out: list[str] = []
    buf = ""
    for clause in _CLAUSE_SPLIT.split(piece):
        if not clause:
            continue
        if len(buf) + len(clause) <= limit:
            buf += clause
            continue
        if buf:
            out.append(buf)
            buf = ""
        if len(clause) <= limit:
            buf = clause
            continue
        # mệnh đề đơn lẻ vẫn quá dài -> cắt theo từ
        line = ""
        for word in clause.split(" "):
            # một "từ" dài hơn cả giới hạn (URL, chuỗi liền không dấu cách):
            # chia thành nhiều mảnh chứ không cắt bỏ phần thừa
            while len(word) > limit:
                if line:
                    out.append(line)
                    line = ""
                out.append(word[:limit])
                word = word[limit:]
            if not word:
                continue
            if len(line) + len(word) + 1 <= limit:
                line = f"{line} {word}".strip()
            else:
                if line:
                    out.append(line)
                line = word
        buf = line
    if buf:
        out.append(buf)
    return [s for s in out if s.strip()]


def split_text(text: str, max_chars: int = 220) -> list[str]:
    """Gom câu thành các đoạn <= max_chars ký tự, giữ nguyên ranh giới câu."""
    text = re.sub(r"[ \t]+", " ", (text or "").strip())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for piece in _hard_wrap(sentence, max_chars):
            if len(buf) + len(piece) + 1 <= max_chars:
                buf = f"{buf} {piece}".strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = piece
    if buf:
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class TTSEngine:
    """Giữ tối đa 1 model trong VRAM. Đổi chế độ = nạp lại checkpoint khác."""

    def __init__(self, device: str | None = None, use_06b: bool = False):
        self.device = device or os.getenv("TTS_DEVICE", "cuda:0")
        self.use_06b = use_06b
        self.model = None
        self.model_id: str | None = None
        self.mode: str | None = None
        # RLock chứ không phải Lock: synthesize gọi ensure_loaded, cả hai cùng khóa.
        self._lock = threading.RLock()

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Không thấy GPU. Kiểm tra driver NVIDIA và bản torch có CUDA "
                "(chạy: python -c \"import torch; print(torch.cuda.is_available())\"). "
                "Muốn chạy thử trên CPU thì đặt TTS_DEVICE=cpu — sẽ rất chậm."
            )

    # -- vòng đời model ----------------------------------------------------- #

    def model_id_for(self, mode: str) -> str:
        if mode not in MODEL_IDS:
            raise ValueError(f"Chế độ không hợp lệ: {mode}")
        if self.use_06b and mode in MODEL_IDS_06B:
            return MODEL_IDS_06B[mode]
        return MODEL_IDS[mode]

    @staticmethod
    def _attn_implementation() -> str:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            return "sdpa"

    def ensure_loaded(self, mode: str) -> str:
        """Nạp model cho `mode`, giải phóng model cũ nếu khác. Trả về dòng trạng thái."""
        from qwen_tts import Qwen3TTSModel

        target = self.model_id_for(mode)
        with self._lock:
            if self.model is not None and self.model_id == target:
                return f"Sẵn sàng · {target} · {self.vram_report()}"

            if self.model is not None:
                log(f"đổi model: {self.model_id} -> {target}")
            self._unload_unlocked()

            on_cpu = self.device.startswith("cpu")
            attn = "eager" if on_cpu else self._attn_implementation()
            log(f"đang nạp {target} trên {self.device} (attn={attn})")
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    target,
                    device_map=self.device,
                    dtype=torch.float32 if on_cpu else torch.bfloat16,
                    attn_implementation=attn,
                )
            except Exception as exc:
                self._unload_unlocked()
                if is_oom(exc):
                    raise OutOfVRAM(
                        "Không đủ VRAM để nạp model. Chạy lại app với --use-06b, "
                        "hoặc thuê GPU nhiều VRAM hơn."
                    ) from exc
                raise

            self.model_id = target
            self.mode = mode
            log(f"đã nạp · {self.vram_report()}")
            return f"Đã nạp · {target} · {self.vram_report()}"

    def _unload_unlocked(self) -> None:
        if self.model is not None:
            del self.model
        self.model = None
        self.model_id = None
        self.mode = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> str:
        with self._lock:
            had = self.model_id
            self._unload_unlocked()
        if had:
            return f"Đã nhả {had} · {self.vram_report()}"
        return f"Chưa có model nào đang nạp · {self.vram_report()}"

    def _device_index(self) -> int | None:
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return None
        _, _, idx = self.device.partition(":")
        return int(idx) if idx.isdigit() else 0

    def vram_report(self) -> str:
        idx = self._device_index()
        if idx is None:
            return "CPU"
        try:
            used = torch.cuda.memory_allocated(idx) / 1024**3
            total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            return f"VRAM {used:.1f}/{total:.0f} GB · {torch.cuda.get_device_name(idx)}"
        except Exception:
            return "VRAM: không đọc được"

    def supported_speakers(self) -> list[str]:
        if self.model is not None and hasattr(self.model, "get_supported_speakers"):
            try:
                found = list(self.model.get_supported_speakers())
                if found:
                    return found
            except Exception:
                pass
        return list(SPEAKERS)

    # -- sinh audio --------------------------------------------------------- #

    def _call(self, fn, texts: list[str], shared: dict, broadcast: dict, gen_kwargs: dict):
        """
        Gọi batch trước cho nhanh; nếu backend không nhận list thì chạy tuần tự.
        `broadcast` là tham số cần nhân bản thành list theo số đoạn.
        `shared` là tham số dùng chung, giữ nguyên (ví dụ voice_clone_prompt).
        """
        n = len(texts)
        if n > 1:
            try:
                batched = {k: (v if isinstance(v, list) else [v] * n) for k, v in broadcast.items()}
                wavs, sr = fn(text=texts, **batched, **shared, **gen_kwargs)
                return list(wavs), sr
            except Exception as exc:
                if is_oom(exc):
                    raise OutOfVRAM(
                        f"Hết VRAM khi đọc {n} đoạn cùng lúc. Giảm 'Ký tự tối đa mỗi đoạn' "
                        "hoặc giảm max_new_tokens rồi thử lại."
                    ) from exc
                log(f"batch thất bại ({type(exc).__name__}: {exc}) — chuyển sang tuần tự")

        wavs: list[np.ndarray] = []
        sr = DEFAULT_SAMPLE_RATE
        for text in texts:
            single = {k: (v[0] if isinstance(v, list) else v) for k, v in broadcast.items()}
            try:
                out, sr = fn(text=text, **single, **shared, **gen_kwargs)
            except Exception as exc:
                if is_oom(exc):
                    raise OutOfVRAM(
                        "Hết VRAM khi sinh audio. Bấm 'Giải phóng VRAM', giảm max_new_tokens, "
                        "hoặc chạy lại app với --use-06b."
                    ) from exc
                raise
            wavs.append(out[0])
        return wavs, sr

    @staticmethod
    def _join(wavs: Iterable[np.ndarray], sr: int) -> np.ndarray:
        arrs = [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs]
        arrs = [a for a in arrs if a.size]
        if not arrs:
            raise RuntimeError("Model không trả về audio nào. Thử rút ngắn văn bản rồi chạy lại.")
        if len(arrs) == 1:
            return arrs[0]
        gap = np.zeros(int(sr * SILENCE_BETWEEN_CHUNKS), dtype=np.float32)
        out: list[np.ndarray] = []
        for i, a in enumerate(arrs):
            if i:
                out.append(gap)
            out.append(a)
        return np.concatenate(out)

    def synthesize(
        self,
        mode: str,
        text: str,
        language: str = "Auto",
        speaker: str | None = None,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        x_vector_only: bool = False,
        max_chars: int = 220,
        max_new_tokens: int = 2048,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> tuple[int, np.ndarray]:
        """Trả về (sample_rate, waveform float32) — đúng định dạng gr.Audio nhận."""

        def step(msg: str) -> None:
            if on_step:
                on_step(msg)

        chunks = split_text(text, max_chars=int(max_chars))
        if not chunks:
            raise ValueError("Chưa có văn bản để đọc.")

        step("Chuẩn bị model")
        self.ensure_loaded(mode)

        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        gen_kwargs: dict = {"max_new_tokens": int(max_new_tokens)}
        if temperature:
            gen_kwargs["temperature"] = float(temperature)
        if top_p:
            gen_kwargs["top_p"] = float(top_p)

        lang = language or "Auto"

        with self._lock:
            if mode == "custom_voice":
                step(f"Đang đọc {len(chunks)} đoạn")
                wavs, sr = self._call(
                    self.model.generate_custom_voice,
                    chunks,
                    shared={},
                    broadcast={
                        "language": lang,
                        "speaker": speaker or "Ryan",
                        "instruct": instruct or "",
                    },
                    gen_kwargs=gen_kwargs,
                )

            elif mode == "voice_design":
                if not (instruct or "").strip():
                    raise ValueError("Chế độ thiết kế giọng cần phần mô tả giọng.")
                step(f"Đang đọc {len(chunks)} đoạn")
                wavs, sr = self._call(
                    self.model.generate_voice_design,
                    chunks,
                    shared={},
                    broadcast={"language": lang, "instruct": instruct},
                    gen_kwargs=gen_kwargs,
                )

            elif mode == "voice_clone":
                if not ref_audio:
                    raise ValueError("Cần một file audio mẫu (3–10 giây) để nhân bản giọng.")
                if not (ref_text or "").strip() and not x_vector_only:
                    raise ValueError(
                        "Cần đúng lời thoại có trong audio mẫu. Nếu không có, "
                        "bật 'chỉ dùng speaker embedding' — độ giống sẽ thấp hơn."
                    )
                step("Đang phân tích giọng mẫu")
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=(ref_text or ""),
                    x_vector_only_mode=bool(x_vector_only),
                )
                step(f"Đang đọc {len(chunks)} đoạn")
                # voice_clone_prompt dùng chung cho mọi đoạn, không nhân bản theo list
                wavs, sr = self._call(
                    self.model.generate_voice_clone,
                    chunks,
                    shared={"voice_clone_prompt": prompt},
                    broadcast={"language": lang},
                    gen_kwargs=gen_kwargs,
                )

            else:
                raise ValueError(f"Chế độ không hợp lệ: {mode}")

        step("Đang ghép audio")
        audio = self._join(wavs, sr)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return sr, audio

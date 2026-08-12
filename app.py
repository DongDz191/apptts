"""
Giao diện web cho Qwen3-TTS 1.7B.

Chạy trên máy vast.ai:
    python app.py --port 7860

Rồi trên Mac mở http://localhost:7860 qua SSH tunnel:
    ssh -p <PORT> root@<HOST> -L 7860:localhost:7860
"""

from __future__ import annotations

import argparse
import functools
import os
import tempfile
import traceback
import zipfile
from pathlib import Path

import gradio as gr
import soundfile as sf

from engine import LANGUAGES, SPEAKERS, OutOfVRAM, TTSEngine, split_text

ENGINE: TTSEngine | None = None

SPEAKER_CHOICES = [f"{name} — {desc}" for name, desc in SPEAKERS.items()]
DEFAULT_SPEAKER = next(c for c in SPEAKER_CHOICES if c.startswith("Ryan"))


def _speaker_id(choice: str) -> str:
    return (choice or "Ryan").split(" — ")[0].strip()


CSS = """
.gradio-container { max-width: 1080px !important; }
#eyebrow {
  font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
  font-size: 12px; letter-spacing: .09em; text-transform: uppercase;
  opacity: .62; margin-bottom: 2px;
}
#status-line textarea {
  font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
  font-size: 12.5px;
}
#title { font-size: 28px; font-weight: 650; letter-spacing: -.02em; margin: 0 0 4px; }
#subtitle { opacity: .72; margin: 0 0 12px; }
.note { font-size: 13px; opacity: .78; line-height: 1.55; }
"""


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #

def guard(fn):
    """Đổi exception thành thông báo đọc được thay vì traceback đỏ trên UI."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except (ValueError, OutOfVRAM) as exc:
            raise gr.Error(str(exc)) from exc
        except FileNotFoundError as exc:
            raise gr.Error(f"Không tìm thấy file: {exc}") from exc
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    return wrapper


def status_now() -> str:
    if ENGINE is None:
        return "Engine chưa khởi tạo."
    return f"{ENGINE.model_id or 'chưa nạp model nào'} · {ENGINE.vram_report()}"


@guard
def do_unload() -> str:
    return ENGINE.unload()


def _stepper(progress: gr.Progress):
    """Nối on_step của engine vào thanh tiến trình của Gradio."""
    state = {"i": 0}
    total = 4

    def on_step(msg: str) -> None:
        state["i"] = min(state["i"] + 1, total)
        progress(state["i"] / total, desc=msg)

    return on_step


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #

@guard
def gen_custom_voice(text, speaker_choice, language, instruct,
                     max_chars, max_new_tokens, temperature, top_p, seed,
                     progress=gr.Progress()):
    sr, audio = ENGINE.synthesize(
        mode="custom_voice",
        text=text,
        language=language,
        speaker=_speaker_id(speaker_choice),
        instruct=(instruct or "").strip() or None,
        max_chars=max_chars,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=int(seed),
        on_step=_stepper(progress),
    )
    return (sr, audio), status_now()


@guard
def gen_voice_design(text, instruct, language,
                     max_chars, max_new_tokens, temperature, top_p, seed,
                     progress=gr.Progress()):
    sr, audio = ENGINE.synthesize(
        mode="voice_design",
        text=text,
        language=language,
        instruct=(instruct or "").strip(),
        max_chars=max_chars,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=int(seed),
        on_step=_stepper(progress),
    )
    return (sr, audio), status_now()


@guard
def gen_voice_clone(text, ref_audio, ref_text, x_vector_only, language,
                    max_chars, max_new_tokens, temperature, top_p, seed,
                    progress=gr.Progress()):
    sr, audio = ENGINE.synthesize(
        mode="voice_clone",
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_text=(ref_text or "").strip() or None,
        x_vector_only=bool(x_vector_only),
        max_chars=max_chars,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=int(seed),
        on_step=_stepper(progress),
    )
    return (sr, audio), status_now()


@guard
def gen_batch(lines_text, speaker_choice, language, instruct, max_new_tokens,
              progress=gr.Progress()):
    lines = [ln.strip() for ln in (lines_text or "").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Mỗi dòng là một file audio. Hiện chưa có dòng nào.")
    if len(lines) > 300:
        raise ValueError("Tối đa 300 dòng mỗi lần chạy. Chia nhỏ ra rồi chạy nhiều lượt.")

    ENGINE.ensure_loaded("custom_voice")

    outdir = Path(tempfile.mkdtemp(prefix="qwen3tts-batch-"))
    made: list[Path] = []
    for i, line in enumerate(progress.tqdm(lines, desc="Đang đọc")):
        sr, audio = ENGINE.synthesize(
            mode="custom_voice",
            text=line,
            language=language,
            speaker=_speaker_id(speaker_choice),
            instruct=(instruct or "").strip() or None,
            max_new_tokens=max_new_tokens,
        )
        path = outdir / f"{i + 1:03d}.wav"
        sf.write(path, audio, sr)
        made.append(path)

    zip_path = outdir / "qwen3-tts-batch.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in made:
            zf.write(p, arcname=p.name)
        zf.writestr(
            "noi_dung.txt",
            "\n".join(f"{i + 1:03d}.wav\t{t}" for i, t in enumerate(lines)),
        )
    return str(zip_path), f"Xong {len(made)} file · {status_now()}"


@guard
def preview_chunks(text, max_chars):
    chunks = split_text(text or "", max_chars=int(max_chars))
    if not chunks:
        return "Chưa có văn bản."
    body = "\n".join(
        f"{i + 1:>2}. ({len(c)} ký tự) {c[:90]}{'…' if len(c) > 90 else ''}"
        for i, c in enumerate(chunks)
    )
    return f"{len(chunks)} đoạn sẽ được đọc rồi ghép lại:\n{body}"


# --------------------------------------------------------------------------- #
# Khối tham số dùng chung
# --------------------------------------------------------------------------- #

def advanced_block():
    with gr.Accordion("Tham số sinh", open=False):
        with gr.Row():
            max_chars = gr.Slider(80, 400, value=220, step=10,
                                  label="Ký tự tối đa mỗi đoạn")
            max_new_tokens = gr.Slider(256, 4096, value=2048, step=128,
                                       label="max_new_tokens")
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05,
                                    label="temperature (0 = mặc định của checkpoint)")
            top_p = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                              label="top_p (0 = mặc định)")
            seed = gr.Number(value=-1, precision=0, label="seed (-1 = ngẫu nhiên)")
    return max_chars, max_new_tokens, temperature, top_p, seed


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def build_ui() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="Qwen3-TTS Studio", theme=theme, css=CSS) as demo:
        gr.HTML(
            '<div id="eyebrow">Qwen3-TTS-12Hz · 1.7B · chạy cục bộ trên GPU thuê</div>'
            '<div id="title">Xưởng giọng nói</div>'
            '<div id="subtitle">Chọn giọng có sẵn, mô tả giọng bằng lời, '
            'hoặc nhân bản giọng từ một đoạn thu 3 giây.</div>'
        )

        with gr.Row():
            status = gr.Textbox(
                label="Trạng thái",
                value="Model nạp khi bạn bấm Đọc lần đầu.",
                interactive=False, elem_id="status-line", scale=5,
            )
            unload_btn = gr.Button("Giải phóng VRAM", scale=1)
        unload_btn.click(do_unload, outputs=status)

        gr.HTML(
            '<p class="note">Model hỗ trợ 10 ngôn ngữ: Trung, Anh, Nhật, Hàn, Đức, Pháp, Nga, '
            'Bồ Đào Nha, Tây Ban Nha, Ý. <b>Tiếng Việt không nằm trong danh sách</b> — chế độ Auto '
            'vẫn đọc được nhưng sai dấu thanh nhiều. Ba tab đầu dùng ba checkpoint khác nhau, '
            'đổi tab sẽ mất 15–30 giây nạp lại.</p>'
        )

        with gr.Tabs():
            # ---------------- Tab 1: giọng có sẵn ---------------- #
            with gr.Tab("Giọng có sẵn"):
                with gr.Row():
                    with gr.Column(scale=3):
                        cv_text = gr.Textbox(
                            label="Văn bản", lines=7,
                            placeholder="She said she would be here by noon.",
                        )
                        cv_instruct = gr.Textbox(
                            label="Chỉ dẫn diễn đạt (tùy chọn)",
                            placeholder="Very happy. / 用特别愤怒的语气说",
                        )
                    with gr.Column(scale=2):
                        cv_speaker = gr.Dropdown(SPEAKER_CHOICES, value=DEFAULT_SPEAKER,
                                                 label="Giọng")
                        cv_lang = gr.Dropdown(LANGUAGES, value="Auto", label="Ngôn ngữ")
                cv_adv = advanced_block()
                cv_btn = gr.Button("Đọc", variant="primary")
                cv_out = gr.Audio(label="Kết quả", type="numpy", show_download_button=True)
                cv_btn.click(
                    gen_custom_voice,
                    [cv_text, cv_speaker, cv_lang, cv_instruct, *cv_adv],
                    [cv_out, status],
                )

            # ---------------- Tab 2: thiết kế giọng ---------------- #
            with gr.Tab("Thiết kế giọng"):
                gr.HTML('<p class="note">Mô tả giọng bằng ngôn ngữ tự nhiên: giới tính, tuổi, '
                        'quãng giọng, cảm xúc, nhịp nói. Càng cụ thể thì giọng càng ổn định.</p>')
                vd_text = gr.Textbox(label="Văn bản", lines=6)
                vd_instruct = gr.Textbox(
                    label="Mô tả giọng", lines=3,
                    placeholder="Male, 17 years old, tenor range, gaining confidence — deeper "
                                "breath support now, though vowels still tighten when nervous",
                )
                vd_lang = gr.Dropdown(LANGUAGES, value="Auto", label="Ngôn ngữ")
                vd_adv = advanced_block()
                vd_btn = gr.Button("Đọc", variant="primary")
                vd_out = gr.Audio(label="Kết quả", type="numpy", show_download_button=True)
                vd_btn.click(
                    gen_voice_design,
                    [vd_text, vd_instruct, vd_lang, *vd_adv],
                    [vd_out, status],
                )

            # ---------------- Tab 3: nhân bản giọng ---------------- #
            with gr.Tab("Nhân bản giọng"):
                gr.HTML('<p class="note">Cần 3–10 giây thu âm sạch, kèm đúng lời thoại trong đoạn '
                        'thu đó. Chỉ nhân bản giọng của chính bạn hoặc giọng bạn có sự đồng ý '
                        'rõ ràng.</p>')
                with gr.Row():
                    with gr.Column():
                        vc_ref = gr.Audio(label="Audio mẫu", type="filepath",
                                          sources=["upload", "microphone"])
                        vc_ref_text = gr.Textbox(label="Lời thoại trong audio mẫu", lines=2)
                        vc_xvec = gr.Checkbox(
                            False,
                            label="Chỉ dùng speaker embedding (không cần lời thoại, giống ít hơn)",
                        )
                    with gr.Column():
                        vc_text = gr.Textbox(label="Văn bản cần đọc", lines=8)
                        vc_lang = gr.Dropdown(LANGUAGES, value="Auto", label="Ngôn ngữ")
                vc_adv = advanced_block()
                vc_btn = gr.Button("Đọc", variant="primary")
                vc_out = gr.Audio(label="Kết quả", type="numpy", show_download_button=True)
                vc_btn.click(
                    gen_voice_clone,
                    [vc_text, vc_ref, vc_ref_text, vc_xvec, vc_lang, *vc_adv],
                    [vc_out, status],
                )

            # ---------------- Tab 4: hàng loạt ---------------- #
            with gr.Tab("Hàng loạt"):
                gr.HTML('<p class="note">Mỗi dòng thành một file .wav riêng, tải về dưới dạng zip. '
                        'Dùng model giọng có sẵn.</p>')
                b_text = gr.Textbox(label="Mỗi dòng một câu", lines=12)
                with gr.Row():
                    b_speaker = gr.Dropdown(SPEAKER_CHOICES, value=DEFAULT_SPEAKER, label="Giọng")
                    b_lang = gr.Dropdown(LANGUAGES, value="Auto", label="Ngôn ngữ")
                b_instruct = gr.Textbox(label="Chỉ dẫn diễn đạt (tùy chọn)")
                b_tokens = gr.Slider(256, 4096, value=2048, step=128, label="max_new_tokens")
                b_btn = gr.Button("Chạy hàng loạt", variant="primary")
                b_out = gr.File(label="Tải về")
                b_btn.click(
                    gen_batch,
                    [b_text, b_speaker, b_lang, b_instruct, b_tokens],
                    [b_out, status],
                )

            # ---------------- Tab 5: kiểm tra cắt đoạn ---------------- #
            with gr.Tab("Kiểm tra cắt đoạn"):
                gr.HTML('<p class="note">Văn bản dài được cắt theo câu rồi ghép audio lại. '
                        'Xem trước để chắc chỗ cắt không rơi vào giữa ý.</p>')
                p_text = gr.Textbox(label="Văn bản", lines=8)
                p_max = gr.Slider(80, 400, value=220, step=10, label="Ký tự tối đa mỗi đoạn")
                p_out = gr.Textbox(label="Các đoạn", lines=12, interactive=False)
                gr.Button("Xem trước").click(preview_chunks, [p_text, p_max], p_out)

    return demo


# --------------------------------------------------------------------------- #
# Khởi động
# --------------------------------------------------------------------------- #

def main():
    global ENGINE

    ap = argparse.ArgumentParser(description="Qwen3-TTS Studio")
    ap.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    ap.add_argument("--share", action="store_true",
                    help="Tạo link gradio.live tạm thời (https, dùng được micro)")
    ap.add_argument("--device", default=os.getenv("TTS_DEVICE", "cuda:0"))
    ap.add_argument("--use-06b", action="store_true",
                    help="Dùng bản 0.6B cho GPU VRAM thấp")
    ap.add_argument("--preload", choices=["custom_voice", "voice_design", "voice_clone"],
                    help="Nạp sẵn một model lúc khởi động")
    ap.add_argument("--ssl-certfile")
    ap.add_argument("--ssl-keyfile")
    args = ap.parse_args()

    ENGINE = TTSEngine(device=args.device, use_06b=args.use_06b)
    print(f"[app] thiết bị: {ENGINE.vram_report()}")

    if args.preload:
        print("[app] " + ENGINE.ensure_loaded(args.preload))

    auth = None
    user, pw = os.getenv("APP_USER"), os.getenv("APP_PASSWORD")
    if user and pw:
        auth = (user, pw)
        print(f"[app] đăng nhập bằng tài khoản: {user}")
    else:
        print("[app] CẢNH BÁO: chưa đặt APP_USER / APP_PASSWORD — trang web mở công khai.")

    print(f"[app] mở trên Mac: http://localhost:{args.port}")
    print(f"[app] nếu chưa có tunnel: ssh -p <PORT> root@<HOST> "
          f"-L {args.port}:localhost:{args.port}")

    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=32)

    launch_kwargs = dict(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        show_error=True,
    )
    if args.ssl_certfile and args.ssl_keyfile:
        launch_kwargs.update(
            ssl_certfile=args.ssl_certfile,
            ssl_keyfile=args.ssl_keyfile,
            ssl_verify=False,
        )
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()

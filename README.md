# Qwen3-TTS Studio — macOS + GPU thuê ở vast.ai

App Python có giao diện web cho **Qwen3-TTS-12Hz-1.7B** (Apache-2.0, Qwen team phát hành 22/01/2026).
Ba chế độ: giọng dựng sẵn, thiết kế giọng bằng mô tả, và nhân bản giọng từ đoạn thu 3 giây.

Mac chỉ đóng vai trò cửa sổ trình duyệt. Toàn bộ model chạy trên máy vast.

```
app.py            giao diện Gradio (5 tab)
engine.py         nạp/đổi model, cắt văn bản dài, gọi 3 hàm sinh của qwen-tts
api_server.py     REST API (tùy chọn)
setup.sh          chạy trên máy vast, một lần — cài đặt + tải trọng số
run.sh            chạy trên máy vast — khởi động app trong tmux
mac.sh            chạy trên Mac — push code, mở tunnel, kéo file về
```

---

## Trước khi bắt đầu: tiếng Việt

Qwen3-TTS hỗ trợ 10 ngôn ngữ — Trung, Anh, Nhật, Hàn, Đức, Pháp, Nga, Bồ Đào Nha, Tây Ban Nha, Ý.
**Tiếng Việt không có trong danh sách.** Đặt ngôn ngữ `Auto` thì model vẫn đọc, nhưng dấu thanh và
nhiều phụ âm cuối sẽ sai — không dùng cho sản phẩm thật được.

Nếu đích đến là tiếng Việt: fine-tune bản Base bằng dữ liệu tiếng Việt (repo Qwen3-TTS có thư mục
`finetuning`), hoặc chọn model đã hỗ trợ sẵn như viXTTS, F5-TTS-Vietnamese. App này vẫn chạy được
với checkpoint đã fine-tune — chỉ cần đổi `MODEL_IDS` trong `engine.py`.

---

## 1. Thuê máy

| Hạng mục | Tối thiểu | Nên dùng |
|---|---|---|
| VRAM | 12 GB | 16–24 GB (RTX 3090 / 4090 / A4000) |
| Disk | 40 GB | 60 GB (image ~20 GB + trọng số ~15 GB + cache) |
| CUDA | 12.1 trở lên | 12.4+ |
| RAM | 16 GB | 32 GB |

Model 1.7B ở bf16 chỉ chiếm ~4 GB trọng số. **Đừng thuê A100/H100** — đắt gấp nhiều lần mà không
nhanh hơn tương ứng vì model quá nhỏ.

Các bước trên vast.ai:

1. Nạp tiền vào tài khoản.
2. Tab **Templates** → chọn image PyTorch có CUDA, ví dụ `pytorch/pytorch` bản `cuda12.4-cudnn9-devel`.
3. Kéo **Disk Space** lên 60 GB **trước khi** thuê. Đổi disk sau khi máy chạy rất phiền.
4. Tab **Search** → lọc GPU, để ý cột `Down` (nên trên 200 Mbps vì phải tải ~15 GB trọng số).
5. Chọn **On-Demand** thay vì Interruptible nếu không muốn bị cướp máy giữa chừng.
6. **Rent**.

Không cần thêm `-p 7860:7860` vào Docker options — hướng dẫn này dùng SSH tunnel, gọn và an toàn hơn.

---

## 2. Sửa lệnh SSH

Vast cho bạn lệnh dạng:

```bash
ssh -p 16885 root@ssh7.vast.ai -L 8080:localhost:8080
```

Cờ `-L 8080` mới chỉ mở đường cho Jupyter. App chạy ở 7860 nên thêm một cờ nữa:

```bash
ssh -p 16885 root@ssh7.vast.ai \
  -L 8080:localhost:8080 \
  -L 7860:localhost:7860 \
  -L 8000:localhost:8000
```

`-L` nối thẳng vào `localhost` bên trong container. Không phơi port ra internet, và trình duyệt coi
`localhost` là nguồn tin cậy nên **micro ở tab nhân bản giọng dùng được**.

Nếu Mac báo `Address already in use`, đổi cổng phía Mac: `-L 7861:localhost:7860` rồi mở `localhost:7861`.

---

## 3. Đẩy code lên (chạy trên Mac)

Mở **cửa sổ Terminal thứ hai**, giữ nguyên cửa sổ SSH kia:

```bash
cd ~/Downloads/qwen3-tts-vastai
VAST_PORT=16885 VAST_HOST=ssh7.vast.ai bash mac.sh push
```

Hoặc làm tay:

```bash
scp -P 16885 -r ~/Downloads/qwen3-tts-vastai root@ssh7.vast.ai:/workspace/
```

`scp` dùng **`-P` viết hoa**, khác `ssh` dùng `-p` thường — đây là lỗi gõ nhầm phổ biến nhất.
Lần đầu kết nối sẽ hỏi `Are you sure you want to continue connecting?` → gõ `yes`.

---

## 4. Cài đặt (trong cửa sổ SSH)

```bash
cd /workspace/qwen3-tts-vastai
bash setup.sh
```

Script kiểm tra GPU trước, dừng ngay nếu `nvidia-smi` hoặc torch-CUDA hỏng — không để bạn đợi
20 phút rồi mới phát hiện máy lỗi. Sau đó cài thư viện và tải ~15 GB trọng số về `/workspace/hf`.

Dòng `flash-attn` có thể fail, không sao — engine tự chuyển sang `sdpa`, chậm hơn chút, kết quả không đổi.

Muốn tiết kiệm disk: mở `setup.sh`, xóa các repo không dùng trong danh sách `repos`.

---

## 5. Chạy

```bash
export APP_USER=admin
export APP_PASSWORD='dat-mat-khau-vao-day'
bash run.sh
```

`run.sh` khởi động app trong `tmux`, nên app vẫn sống khi Mac sleep hoặc SSH đứt.

```bash
bash run.sh logs    # xem log; thoát ra bằng Ctrl+B rồi D
bash run.sh stop    # dừng hẳn
```

Trên Mac, `Ctrl` ở đây là phím `Control` thật, không phải `Command`.

Không đặt `APP_PASSWORD` thì app mở công khai — trên máy thuê công cộng, đừng bỏ qua.

---

## 6. Mở giao diện

Trên Mac: **http://localhost:7860**

Đợi dòng `Running on local URL` hiện trong log rồi hãy mở; mở sớm quá sẽ thấy trang trắng.

Mất kết nối do Mac sleep hay đổi wifi? App vẫn chạy trong tmux. Chỉ cần chạy lại lệnh SSH ở bước 2
là vào được ngay, không phải cài hay chạy lại gì.

Muốn tunnel chạy ngầm không chiếm cửa sổ Terminal:

```bash
bash mac.sh tunnel      # hoặc: ssh -N -f -p 16885 root@ssh7.vast.ai -L 7860:localhost:7860
pkill -f "ssh -N -f -p 16885"   # tắt khi xong
```

**Cách thay thế:** `python app.py --share` cho một link `*.gradio.live` chạy 72 giờ, dùng được HTTPS
và micro, gửi cho người khác xem thử được. Nhưng ai có link cũng vào được — nhớ đặt mật khẩu.

---

## 7. REST API (tùy chọn)

Trên máy vast:

```bash
export API_KEY='mot-chuoi-bi-mat'
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Trên Mac (đã có tunnel 8000):

```bash
curl -X POST http://localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: mot-chuoi-bi-mat' \
  -d '{"mode":"custom_voice","text":"She said she would be here by noon.","speaker":"Ryan","language":"English"}' \
  --output out.wav
open out.wav
```

`GET /healthz` xem model nào đang nạp. `GET /voices` liệt kê giọng và ngôn ngữ. `POST /unload` nhả VRAM.

---

## 8. Ba chế độ

| Tab | Checkpoint | Dùng khi |
|---|---|---|
| Giọng có sẵn | `1.7B-CustomVoice` | Cần giọng ổn định, lặp lại được. 9 giọng, thêm chỉ dẫn cảm xúc bằng câu lệnh. |
| Thiết kế giọng | `1.7B-VoiceDesign` | Cần nhân vật cụ thể mà không có mẫu thu. Mô tả bằng lời: giới tính, tuổi, quãng giọng, tâm trạng. |
| Nhân bản giọng | `1.7B-Base` | Có 3–10 giây thu âm sạch và biết chính xác lời thoại trong đó. |

Ba checkpoint khác nhau, **không nạp đồng thời được** trên một GPU nhỏ. Đổi tab thì engine tự nhả
model cũ rồi nạp model mới, mất 15–30 giây.

Mẹo giữ nhân vật nhất quán qua nhiều câu: dùng tab Thiết kế giọng tạo một đoạn mẫu ưng ý, tải về,
rồi nạp lại nó làm audio mẫu ở tab Nhân bản giọng.

**Về nhân bản giọng:** chỉ dùng giọng của chính bạn hoặc giọng bạn có sự đồng ý rõ ràng. Giả giọng
người khác để lừa đảo hoặc bôi nhọ là vi phạm pháp luật ở phần lớn các nước, kể cả Việt Nam.

---

## 9. Lấy file về và dọn dẹp

```bash
bash mac.sh pull     # kéo .wav về ~/Downloads/qwen3-tts-out
```

Bấm **Stop** trên console vast.ai khi nghỉ — nhưng instance đã dừng **vẫn tính tiền disk** mỗi giờ.
Xong việc hẳn thì **Destroy**. Nhớ tải file về trước, disk mất theo instance.

---

## 10. Trục trặc thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `scp: Connection refused` | Dùng `-P` viết hoa cho scp, `-p` thường cho ssh. |
| `Address already in use` (trên Mac) | Cổng 7860 đang bận. Đổi `-L 7861:localhost:7860`, mở `localhost:7861`. |
| `channel: open failed: connect failed` | App chưa chạy. Vào SSH kiểm tra `bash run.sh logs`. |
| `CUDA out of memory` | Bấm **Giải phóng VRAM**, giảm max_new_tokens, hoặc chạy `EXTRA_ARGS=--use-06b bash run.sh`. |
| `torch.cuda.is_available()` là False | Image thiếu bản torch GPU. Cài lại torch đúng bản CUDA rồi chạy lại setup.sh. |
| `flash-attn` build lỗi | Bỏ qua, engine tự dùng `sdpa`. |
| Tải model rất chậm | Máy mạng yếu. Destroy và thuê máy khác có `Down` cao hơn. |
| Nút micro bị mờ | Bạn đang vào qua IP công khai (HTTP). Dùng tunnel rồi mở `localhost`. |
| Mất app khi đóng Terminal | Phải chạy qua `run.sh` (tmux), không chạy `python app.py` trực tiếp. |
| Câu dài bị cụt | Tăng `max_new_tokens`, hoặc giảm "Ký tự tối đa mỗi đoạn" xuống 150. |

---

## Tham khảo

- Repo gốc: <https://github.com/QwenLM/Qwen3-TTS> (Apache-2.0)
- Trọng số: <https://huggingface.co/collections/Qwen/qwen3-tts>
- Vast.ai networking: <https://docs.vast.ai/networking>

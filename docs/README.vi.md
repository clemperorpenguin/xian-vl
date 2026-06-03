# Xian-MAGE — Trợ lý Thị giác-Ngôn ngữ Thời gian thực cho Môi trường Game 🧙‍♂️🎮🤖

Xian-MAGE là một HUD và trợ lý chơi game trên màn hình nền tảng Linux (Wayland) hoạt động theo thời gian thực và liên tục. Được hỗ trợ bởi backend **[Lemonade Server](https://lemonade-server.ai/)**, nó phủ lên trên các môi trường game đang hoạt động các tính năng OCR thời gian thực, dịch thuật, trực quan hóa cú nhấp chuột định vị hình ảnh và trò chuyện hội thoại tương tác.

Vì quá trình suy luận được điều phối qua Lemonade, **MAGE hỗ trợ thực thi tăng tốc Vulkan, chạy mượt mà trên GPU AMD Radeon™ và các bộ tăng tốc khác.**

Xem thử nghiệm trên YouTube: https://www.youtube.com/watch?v=Izu_8pql7cE

<img width="400" height="340" alt="mage" src="https://github.com/user-attachments/assets/bb51b2c6-378f-4a3e-b25d-05ad284e374b" />

---

## 🌟 Tính năng Cốt lõi (Hoạt động đầy đủ)

- **Tăng tốc Vulkan / GPU AMD**: Thực thi Mô hình Ngôn ngữ-Thị giác với độ trễ thấp được hỗ trợ bởi tăng tốc phần cứng GPU backend.
- **Lớp phủ Desktop cho phép nhấp xuyên qua**: Các cửa sổ lớp phủ PyQt6 trong suốt hiển thị văn bản đã dịch trực tiếp trên HUD và hội thoại game, trong khi vẫn hoàn toàn vô hình với các đầu vào của chuột.
- **Phím nóng Toàn cục Wayland & OSD Lệnh**: Kích hoạt dịch thuật, cấu hình OSD và thanh bên một cách liền mạch bằng các phím chủ đạo toàn hệ thống có thể tùy chỉnh.
- **Chế độ Hội thoại (Tự động chạy VN / Cốt truyện RPG)**: Khóa vào một vùng màn hình, tự động dịch và đẩy làm mới bản dịch trực tiếp chỉ bằng một cú nhấp chuột đơn giản.
- **Làm nổi bật Mục tiêu Định vị Trực quan**: Hỏi trợ lý *"Tôi nhấp vào đâu?"* hoặc *"Lối ra ở đâu?"* và xem nó làm nổi bật tọa độ vật lý chính xác trên màn hình của bạn.
- **Chế độ Điện ảnh (Dịch giọng nói theo ngữ cảnh)**: Kết hợp liền mạch phân tích tầm nhìn chụp màn hình với dịch thuật phát lại âm thanh.
- **Từ điển CC-CEDICT Cục bộ**: Di chuột qua bất kỳ bong bóng dịch nào và nhấn `Alt` để phân tích cú pháp an toàn theo luồng cục bộ các ký tự Trung Quốc, bính âm và định nghĩa.

---

## 🛠️ Bắt đầu (Máy khách MAGE)

### Yêu cầu

- **Linux với Wayland** (cũng hỗ trợ X11; liên kết phím toàn cục yêu cầu đầu vào `evdev`)
- **Quyền người dùng**: Người dùng của bạn phải nằm trong nhóm `input` để bắt phím nóng toàn cục (`sudo usermod -aG input $USER` và đăng xuất/đăng nhập lại)
- **Lemonade Server**: Một phiên bản Lemonade Server đang chạy (mặc định có thể truy cập tại `http://localhost:13305`)

### Cài đặt nhanh (Linux)

Sao chép kho lưu trữ và chạy tập lệnh bootstrap — nó sẽ cài đặt [`uv`](https://docs.astral.sh/uv/), đồng bộ hóa tất cả các phụ thuộc và tự động khởi chạy MAGE:

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
./mage.sh
```

Để thêm MAGE vào menu ứng dụng desktop của bạn:

```bash
./mage.sh --install
```

Để thêm MAGE vào menu của bạn **và** tự động cài đặt các phụ thuộc hệ thống, xây dựng máy chủ Lemonade có thể nhúng từ mã nguồn và kéo xuống mô hình ngôn ngữ-thị giác mặc định:

```bash
./mage.sh --install --build
```

Để xóa mục desktop và biểu tượng:

```bash
./mage.sh --uninstall
```

### Bản phát hành dựng sẵn (Windows & macOS)

Nếu bạn đang chạy Windows hoặc macOS, hoặc không muốn xây dựng từ mã nguồn, hãy tải xuống các gói dựng sẵn từ trang [GitHub Releases](https://github.com/clemperorpenguin/xian-vl/releases).

Các bản phát hành có hai biến thể:
- **Lite**: Phiên bản nhẹ độc lập. Yêu cầu kết nối với Lemonade Server đang chạy bên ngoài.
- **Full**: Đi kèm với máy chủ `lemond` được nhúng, tự động khởi động và dừng khi ứng dụng chạy.

#### Windows
1. Tải xuống `mage-client-Windows-x86-64-lite.zip` hoặc `mage-client-Windows-x86-64-full.zip`.
2. Giải nén tệp lưu trữ.
3. Nhấp đúp vào `mage-client.exe` để chạy.

#### macOS
1. Tải xuống `mage-client-MacOS-ARM64-lite.dmg` hoặc `mage-client-MacOS-ARM64-full.dmg` (hoặc các tệp ZIP tương đương).
2. Nhấp đúp vào DMG và kéo `mage-client.app` vào thư mục **Applications** của bạn.
3. Mở và chạy ứng dụng.
   > [!NOTE]
   > Vì ứng dụng không được Apple chứng nhận/ký, Gatekeeper sẽ chặn nó với cảnh báo ứng dụng bị "hỏng và không thể mở được". Bạn có thể dễ dàng khắc phục điều này bằng cách chạy lệnh sau trong terminal:
   > ```bash
   > xattr -cr /Applications/mage-client.app
   > ```

### Cài đặt thủ công (Tất cả nền tảng)

Nếu bạn thích tự quản lý môi trường:

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
uv sync --all-packages
uv run --package mage-client mage
```

### Điều khiển
- **Mở Menu Hành động / OSD** — Nhấp đúp `Shift` (Phím chủ đạo Mặc định)
- **Kích hoạt Chụp Màn hình** — `C` trong Menu Hành động (chọn vùng màn hình, sau đó Dịch / Hội thoại / Trò chuyện)
- **Chuyển đổi Thanh bên Trò chuyện** — `A` trong Menu Hành động
- **Dịch cho Trò chuyện (Đầu vào)** — `T` trong Menu Hành động
- **Bảng Cài đặt** — `S` trong Menu Hành động

---

## 📁 Kiến trúc & Các dự án vệ tinh

Kho lưu trữ đơn (monorepo) chứa máy khách MAGE cốt lõi sẵn sàng cho sản xuất cũng như các khung dự án đồng hành thử nghiệm:

```
├── apps/
│   ├── mage-client/      # 🧙‍♂️ Ứng dụng HUD chơi game chính đã được xác minh, dựa trên PyQt6
│   ├── nate/             # 📱 Trình đọc OCR và từ điển đồng hành cho Android (Thử nghiệm)
│   ├── masha-extension/  # 🌐 Tiện ích mở rộng trình duyệt dịch văn bản được chọn (Thử nghiệm)
│   ├── lore-client/      # 📜 CLI xây dựng wiki kiến thức RAG (Thử nghiệm)
│   └── luduan-client/    # 🦤 CLI dịch EPUB và sách nói (Thử nghiệm)
└── packages/
    ├── xian-vl/          # ⚙️ Công cụ điều phối LLM/ASR cốt lõi & trình quản lý ngữ cảnh
    └── shared-types/     # 📦 Các mô hình chuẩn, hằng số và kiểu dữ liệu dùng chung
```

> [!WARNING]
> **Hạn chế ASR / Âm thanh**: Việc tải lên luồng âm thanh trực tiếp tới Lemonade hiện đang bị lỗi do các hạn chế của backend phía máy chủ. Do đó, các tính năng dịch giọng nói trực tiếp (như **Chế độ Đột kích**) đã bị tắt. Tất cả các tính năng OCR hình ảnh, dịch văn bản, từ điển và định vị trò chuyện hoạt động hoàn toàn bình thường.

---

## 📜 Giấy phép

Dự án này được cấp phép theo Giấy phép Công cộng GNU phiên bản 3.0. Xem tệp [LICENSE](LICENSE) để biết chi tiết.

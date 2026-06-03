# Xian-MAGE — 游戏环境实时视觉语言助手 🧙‍♂️🎮🤖

Xian-MAGE 是一款面向 Linux (Wayland) 的实时、常驻桌面游戏 HUD 与助手。由 **[Lemonade Server](https://lemonade-server.ai/)** 后端提供支持，它可以将实时 OCR、翻译、视觉定位点击效果可视化以及交互式对话聊天直接叠加在活动游戏画面之上。

由于推理过程由 Lemonade 进行编排，**MAGE 支持 Vulkan 加速执行，可在 AMD Radeon™ GPU 和其他加速器上流畅运行。**

在 YouTube 上观看实际效果：https://www.youtube.com/watch?v=Izu_8pql7cE

<img width="400" height="340" alt="mage" src="https://github.com/user-attachments/assets/bb51b2c6-378f-4a3e-b25d-05ad284e374b" />

---

## 🌟 核心功能（全面可用）

- **Vulkan / AMD GPU 加速**：由后端 GPU 硬件加速提供动力，实现低延迟的视觉语言模型执行。
- **支持点击穿透的桌面悬浮窗**：透明的 PyQt6 悬浮窗，可直接在游戏 HUD 和对话框上方显示翻译文本，同时完全不影响鼠标点击输入。
- **Wayland 全局热键与命令 OSD**：使用可定制的系统级引导热键，无缝触发翻译、OSD 配置和侧边栏。
- **对话模式（自动推视觉小说 / 剧情 RPG）**：锁定屏幕区域并自动翻译，只需简单点击鼠标即可在悬浮窗内推进/刷新翻译。
- **视觉定位目标高亮**：询问助手 *"我该点哪里？"* 或 *"出口在哪里？"*，看着它在屏幕上高亮显示精确的物理坐标。
- **电影模式（上下文语音翻译）**：将屏幕捕获视觉分析与音频播放翻译无缝结合。
- **本地 CC-CEDICT 词典**：将鼠标悬停在任意翻译气泡上并按住 `Alt` 键，即可在本地进行线程安全的汉字、拼音和释义解析。

---

## 🛠️ 快速开始（MAGE 客户端）

### 环境要求

- **支持 Wayland 的 Linux**（也支持 X11；全局键绑定需要 `evdev` 输入）
- **用户权限**：您的用户必须位于 `input` 组中才能捕获全局热键（执行 `sudo usermod -aG input $USER` 并注销/重新登录）
- **Lemonade Server**：运行中的 Lemonade Server 实例（默认通过 `http://localhost:13305` 访问）

### 快速设置（Linux）

克隆仓库并运行引导脚本——它将安装 [`uv`](https://docs.astral.sh/uv/)，同步所有依赖项，并自动启动 MAGE：

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
./mage.sh
```

将 MAGE 添加到桌面应用菜单：

```bash
./mage.sh --install
```

将 MAGE 添加到菜单**并**自动安装系统依赖项，从源码构建可嵌入的 Lemonade 服务器，并拉取默认的视觉语言模型：

```bash
./mage.sh --install --build
```

移除桌面条目和图标：

```bash
./mage.sh --uninstall
```

### 预编译版本（Windows 和 macOS）

如果您使用的是 Windows 或 macOS，或者不想从源码构建，请从 [GitHub Releases](https://github.com/clemperorpenguin/xian-vl/releases) 页面下载预编译包。

发布版本分为两种变体：
- **Lite（精简版）**：独立的轻量级版本。需要连接到外部运行的 Lemonade Server。
- **Full（完整版）**：内置嵌入式 `lemond` 服务器，该服务器会在应用程序运行时自动启动和停止。

#### Windows
1. 下载 `mage-client-Windows-x86-64-lite.zip` 或 `mage-client-Windows-x86-64-full.zip`。
2. 解压压缩包。
3. 双击 `mage-client.exe` 运行。

#### macOS
1. 下载 `mage-client-MacOS-ARM64-lite.dmg` 或 `mage-client-MacOS-ARM64-full.dmg`（或等效的 ZIP 压缩包）。
2. 双击 DMG 文件并将 `mage-client.app` 拖拽到**应用程序**目录。
3. 打开并运行应用程序。
   > [!NOTE]
   > 由于该应用未经 Apple 公证/签名，Gatekeeper 会拦截它并提示应用“已损坏，无法打开”。您可以在终端中运行以下命令来轻松解决此问题：
   > ```bash
   > xattr -cr /Applications/mage-client.app
   > ```

### 手动设置（所有平台）

如果您希望自行管理环境：

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
uv sync --all-packages
uv run --package mage-client mage
```

### 控制操作
- **打开操作菜单 / OSD** — 双击 `Shift`（默认引导键）
- **触发屏幕捕获** — 操作菜单 `C`（选择屏幕区域，然后选择 翻译 / 对话 / 聊天）
- **切换聊天侧边栏** — 操作菜单 `A`
- **为聊天翻译（输入）** — 操作菜单 `T`
- **设置面板** — 操作菜单 `S`

---

## 📁 架构与分支项目

该单仓（monorepo）包含核心的生产就绪 MAGE 客户端以及实验性的周边项目脚手架：

```
├── apps/
│   ├── mage-client/      # 🧙‍♂️ 主验证过的、基于 PyQt6 的游戏 HUD 应用程序
│   ├── nate/             # 📱 Android 配套 OCR 阅读器和词典（实验性）
│   ├── masha-extension/  # 🌐 浏览器扩展选择翻译器（实验性）
│   ├── lore-client/      # 📜 RAG 知识库构建器 CLI（实验性）
│   └── luduan-client/    # 🦤 EPUB 翻译和有声书 CLI（实验性）
└── packages/
    ├── xian-vl/          # ⚙️ 核心 LLM/ASR 编排引擎和上下文管理器
    └── shared-types/     # 📦 规范模型、常量和共享类型
```

> [!WARNING]
> **ASR / 音频限制**：由于服务器端后端限制，目前向 Lemonade 发送实时音频流上传的功能已损坏。因此，实时语音翻译功能（如**突袭模式**）已被禁用。所有视觉 OCR、文本翻译、词典和聊天定位功能均可正常使用。

---

## 📜 许可证

本项目采用 GNU 通用公共许可证 v3.0 授权。详情请参阅 [LICENSE](LICENSE) 文件。

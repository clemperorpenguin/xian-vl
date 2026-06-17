#!/usr/bin/env bash
# MAGE — Gaming HUD for real-time screen translation.
# Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)

# ──────────────────────────────────────────────────────────────────────────────
# mage.sh — Cross-platform bootstrap for MAGE on Linux and macOS.
#
#   ./mage.sh              Install uv + deps, then launch MAGE.
#   ./mage.sh --install    Register MAGE in your application menu / Launchpad.
#   ./mage.sh --install --build
#                          Register MAGE, install embeddable Lemonade, and pull
#                          the default vision-language model.
#   ./mage.sh --uninstall  Remove the application entry created by --install.
#   ./mage.sh --help       Show this help message.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICON_SRC="${REPO_DIR}/xian.png"
PLATFORM="$(uname -s)"

# Linux desktop integration paths
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/512x512/apps"
DESKTOP_FILE="${DESKTOP_DIR}/mage.desktop"
ICON_FILE="${ICON_DIR}/mage.png"

# macOS application bundle path
MACOS_APP="/Applications/MAGE.app"

# ── Helpers ───────────────────────────────────────────────────────────────────

print_usage() {
    cat <<EOF
Usage: ./mage.sh [OPTION]

Options:
  --install     Register MAGE in your application menu (Linux) or Launchpad
                (macOS). On macOS, embeddable Lemonade is downloaded and the
                default model is pulled automatically.
  --install --build
                Register MAGE and build embeddable Lemonade from source
                instead of downloading a pre-built binary.
  --uninstall   Remove the application entry created by --install.
  --build       Build embeddable Lemonade from source without registering.
  --help        Show this help message.

With no options, MAGE is launched directly. If uv or project dependencies
are missing they are installed automatically on first run.
EOF
}

ensure_uv() {
    if command -v uv &>/dev/null; then
        return
    fi

    echo "── uv not found. Installing via astral.sh installer… ──"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # The installer adds uv to ~/.local/bin (or ~/.cargo/bin). Source the env
    # file if it exists, otherwise extend PATH manually.
    if [[ -f "${HOME}/.local/bin/env" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/.local/bin/env"
    elif [[ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]]; then
        # shellcheck source=/dev/null
        source "${CARGO_HOME:-$HOME/.cargo}/env"
    else
        export PATH="${HOME}/.local/bin:${PATH}"
    fi

    if ! command -v uv &>/dev/null; then
        echo "Error: uv installation succeeded but 'uv' is not on PATH." >&2
        echo "Add ~/.local/bin to your PATH and try again." >&2
        exit 1
    fi

    echo "── uv installed successfully. ──"
}

sync_deps() {
    echo "── Syncing workspace dependencies… ──"
    (cd "${REPO_DIR}" && uv sync --all-packages)
}

# ── Linux-only: build dependencies for compiling Lemonade from source ────────

install_build_deps() {
    echo "── Detecting Linux distribution family… ──"
    local distro=""
    if [[ -f /etc/os-release ]]; then
        # Source /etc/os-release in a subshell to keep the variables local
        distro=$(sh -c '
            . /etc/os-release
            ID_NORM=$(echo "${ID:-}" | tr "[:upper:]" "[:lower:]")
            ID_LIKE_NORM=$(echo "${ID_LIKE:-}" | tr "[:upper:]" "[:lower:]")
            
            if [[ "$ID_NORM" =~ (ubuntu|debian|pop|mint) ]] || [[ "$ID_LIKE_NORM" =~ (ubuntu|debian) ]]; then
                echo "debian"
            elif [[ "$ID_NORM" =~ (fedora|rhel|centos|rocky|almalinux) ]] || [[ "$ID_LIKE_NORM" =~ (fedora|rhel) ]]; then
                echo "fedora"
            elif [[ "$ID_NORM" =~ (opensuse|sles) ]] || [[ "$ID_LIKE_NORM" =~ (suse|opensuse) ]]; then
                echo "opensuse"
            elif [[ "$ID_NORM" =~ (arch|manjaro) ]] || [[ "$ID_LIKE_NORM" =~ (arch) ]]; then
                echo "arch"
            else
                echo "unknown"
            fi
        ')
    fi

    echo "Detected distro family: ${distro}"

    case "${distro}" in
        debian)
            echo "Installing build dependencies via apt-get…"
            sudo apt-get update
            sudo apt-get install -y cmake ninja-build g++ pkg-config libssl-dev libdrm-dev
            ;;
        fedora)
            echo "Installing build dependencies via dnf…"
            sudo dnf install -y cmake ninja-build gcc-c++ pkgconf-pkg-config openssl-devel libdrm-devel
            ;;
        opensuse)
            echo "Installing build dependencies via zypper…"
            sudo zypper in -y cmake ninja gcc-c++ pkg-config libopenssl-devel libdrm-devel
            ;;
        arch)
            echo "Installing build dependencies via pacman…"
            sudo pacman -S --needed --noconfirm base-devel cmake ninja pkgconf openssl libdrm
            ;;
        *)
            echo "Warning: Could not automatically detect a supported distro family (Ubuntu/Debian, Fedora/RHEL, openSUSE, or Arch-based)." >&2
            echo "Please manually install the equivalent packages for: cmake, ninja, g++, pkg-config, openssl-dev, libdrm-dev." >&2
            read -p "Press Enter to attempt building anyway, or Ctrl+C to abort."
            ;;
    esac
}

# ── Linux: build Lemonade from source ────────────────────────────────────────

build_lemonade() {
    echo "── Building embeddable Lemonade from source… ──"
    
    # Initialize/update submodule if CMakeLists.txt is missing
    if [[ ! -f "${REPO_DIR}/lemonade/CMakeLists.txt" ]]; then
        echo "Initializing lemonade submodule…"
        git submodule update --init --recursive lemonade
    fi
    
    # Configure and build
    (
        cd "${REPO_DIR}/lemonade"
        echo "Configuring CMake…"
        cmake --preset default -DBUILD_WEB_APP=OFF
        echo "Building target: embeddable…"
        cmake --build --preset default --target embeddable
    )
    
    # Locate built staging directory
    local glob_pattern="${REPO_DIR}/lemonade/build/lemonade-embeddable-*"
    # Expand glob to array
    local dirs=($glob_pattern)
    if [[ ${#dirs[@]} -eq 0 || ! -d "${dirs[0]}" ]]; then
        echo "Error: Could not find built embeddable directory at ${glob_pattern}." >&2
        exit 1
    fi
    local stage_dir="${dirs[0]}"
    echo "Staged files located at: ${stage_dir}"
    
    _install_lemonade_from_dir "${stage_dir}"
}

# ── macOS: download pre-built Lemonade from GitHub releases ──────────────────

download_lemonade() {
    echo "── Downloading pre-built embeddable Lemonade for macOS… ──"

    # Determine latest release tag
    local release_json
    release_json=$(curl -fsSL "https://api.github.com/repos/lemonade-sdk/lemonade/releases/latest")
    local tag
    tag=$(echo "${release_json}" | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
    local version="${tag#v}"
    echo "Latest Lemonade release: ${version}"

    # Determine architecture suffix
    local arch
    arch="$(uname -m)"
    local arch_suffix="macos-arm64"
    if [[ "${arch}" == "x86_64" ]]; then
        arch_suffix="macos-x64"
    fi

    local tarball="lemonade-embeddable-${version}-${arch_suffix}.tar.gz"
    local download_url="https://github.com/lemonade-sdk/lemonade/releases/download/${tag}/${tarball}"

    local tmp_dir
    tmp_dir=$(mktemp -d)
    trap "rm -rf '${tmp_dir}'" EXIT

    echo "Downloading ${download_url}…"
    curl -fSL -o "${tmp_dir}/${tarball}" "${download_url}"

    echo "Extracting…"
    tar -xzf "${tmp_dir}/${tarball}" -C "${tmp_dir}"

    # Find the extracted directory (lemonade-embeddable-*)
    local extracted_dir
    extracted_dir=$(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [[ -z "${extracted_dir}" || ! -d "${extracted_dir}" ]]; then
        echo "Error: Could not find extracted lemonade directory in ${tmp_dir}." >&2
        exit 1
    fi

    _install_lemonade_from_dir "${extracted_dir}"

    # Clean up trap will remove tmp_dir
    trap - EXIT
    rm -rf "${tmp_dir}"
}

# ── Shared: install lemonade binaries from a staging directory ────────────────

_install_lemonade_from_dir() {
    local stage_dir="$1"

    # Copy lemond server binary
    echo "Copying lemond server to repository root…"
    cp -f "${stage_dir}/lemond" "${REPO_DIR}/lemond"
    chmod +x "${REPO_DIR}/lemond"
    
    # Copy resources
    echo "Copying resources to repository root…"
    if [[ -d "${REPO_DIR}/resources" ]]; then
        rm -rf "${REPO_DIR}/resources"
    fi
    cp -r "${stage_dir}/resources" "${REPO_DIR}/resources"
    
    # Copy lemonade CLI
    echo "Copying lemonade CLI to ~/.local/bin/lemonade…"
    mkdir -p "${HOME}/.local/bin"
    cp -f "${stage_dir}/lemonade" "${HOME}/.local/bin/lemonade"
    chmod +x "${HOME}/.local/bin/lemonade"
    
    echo "✓ Lemonade binaries and resources installed successfully."
    
    # Start lemond in background to allow pulling model
    echo "Starting lemond daemon in background to pull model…"
    local log_dir="${HOME}/.cache/lemonade"
    mkdir -p "${log_dir}"
    
    # Launch lemond using absolute path and its resources from the repo root
    pushd "${REPO_DIR}" >/dev/null
    ./lemond --port 13305 > "${log_dir}/install_lemond.log" 2>&1 &
    local lemond_pid=$!
    popd >/dev/null
    
    # Wait for lemond to be ready
    echo "Waiting for lemond to respond on port 13305…"
    local ready=false
    for i in {1..30}; do
        if curl -sf "http://127.0.0.1:13305/v1/models" >/dev/null 2>&1; then
            ready=true
            break
        fi
        sleep 0.5
    done
    
    if [[ "${ready}" != "true" ]]; then
        echo "Error: lemond server failed to start. Log output:" >&2
        cat "${log_dir}/install_lemond.log" >&2
        kill "${lemond_pid}" 2>/dev/null || true
        exit 1
    fi
    
    echo "lemond daemon is ready. Pulling model LMX-Omni-5.5B-Lite…"
    if ! "${HOME}/.local/bin/lemonade" pull LMX-Omni-5.5B-Lite; then
        echo "Error: Failed to pull model LMX-Omni-5.5B-Lite." >&2
        kill "${lemond_pid}" 2>/dev/null || true
        exit 1
    fi
    
    echo "Stopping background lemond daemon…"
    kill "${lemond_pid}" 2>/dev/null || true
    wait "${lemond_pid}" 2>/dev/null || true
    
    echo "✓ Model LMX-Omni-5.5B-Lite pulled and cached successfully."
}

# ── Actions ───────────────────────────────────────────────────────────────────

do_install_linux() {
    ensure_uv
    sync_deps

    # Install icon
    mkdir -p "${ICON_DIR}"
    if [[ -f "${ICON_SRC}" ]]; then
        cp "${ICON_SRC}" "${ICON_FILE}"
        echo "Icon installed → ${ICON_FILE}"
    else
        echo "Warning: ${ICON_SRC} not found; desktop entry will have no icon." >&2
    fi

    # Generate desktop entry
    mkdir -p "${DESKTOP_DIR}"
    cat > "${DESKTOP_FILE}" <<DESKTOP
[Desktop Entry]
Name=MAGE
Comment=MAGE — Gaming HUD for real-time screen translation
Exec=${REPO_DIR}/mage.sh %U
Icon=mage
Type=Application
Categories=Game;Utility;
Terminal=false
DESKTOP

    # Validate the desktop file if desktop-file-validate is available
    if command -v desktop-file-validate &>/dev/null; then
        if desktop-file-validate "${DESKTOP_FILE}" 2>/dev/null; then
            echo "Desktop file validated successfully."
        else
            echo "Warning: desktop-file-validate reported issues (non-fatal)." >&2
        fi
    fi

    # Update the desktop database if available
    if command -v update-desktop-database &>/dev/null; then
        update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true
    fi

    echo ""
    echo "✓ MAGE desktop entry installed."
    echo "  Desktop file → ${DESKTOP_FILE}"
    echo "  You should now see MAGE in your application menu."
    echo "  To launch from the terminal: ${REPO_DIR}/mage.sh"

    # Build and install lemonade if requested
    if [[ "${BUILD_LEMONADE}" == "true" ]]; then
        install_build_deps
        build_lemonade
    fi
}

do_install_macos() {
    ensure_uv
    sync_deps

    echo "── Creating macOS application bundle… ──"

    local app_dir="${MACOS_APP}"
    local contents="${app_dir}/Contents"
    local macos_dir="${contents}/MacOS"
    local res_dir="${contents}/Resources"

    mkdir -p "${macos_dir}" "${res_dir}"

    # Generate .icns icon from xian.png if sips is available
    if [[ -f "${ICON_SRC}" ]] && command -v sips &>/dev/null && command -v iconutil &>/dev/null; then
        echo "Generating macOS icon from xian.png…"
        local iconset_dir
        iconset_dir=$(mktemp -d)/mage.iconset
        mkdir -p "${iconset_dir}"

        # Generate all required icon sizes
        for size in 16 32 64 128 256 512; do
            sips -z "${size}" "${size}" "${ICON_SRC}" --out "${iconset_dir}/icon_${size}x${size}.png" >/dev/null 2>&1
        done
        # Retina variants (e.g., icon_16x16@2x.png = 32x32)
        for size in 16 32 128 256; do
            local double=$((size * 2))
            cp "${iconset_dir}/icon_${double}x${double}.png" "${iconset_dir}/icon_${size}x${size}@2x.png" 2>/dev/null || true
        done
        # 512@2x = 1024, generate separately
        sips -z 1024 1024 "${ICON_SRC}" --out "${iconset_dir}/icon_512x512@2x.png" >/dev/null 2>&1

        iconutil -c icns -o "${res_dir}/mage.icns" "${iconset_dir}" 2>/dev/null || true
        rm -rf "$(dirname "${iconset_dir}")"
    elif [[ -f "${ICON_SRC}" ]]; then
        # Fallback: just copy the PNG
        cp "${ICON_SRC}" "${res_dir}/mage.png"
    fi

    # Determine the icon filename that was created
    local icon_file="mage.icns"
    if [[ ! -f "${res_dir}/mage.icns" ]]; then
        icon_file="mage.png"
    fi

    # Create Info.plist
    cat > "${contents}/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>MAGE</string>
    <key>CFBundleDisplayName</key>
    <string>MAGE</string>
    <key>CFBundleIdentifier</key>
    <string>systems.pendragon.mage</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleExecutable</key>
    <string>mage-launcher</string>
    <key>CFBundleIconFile</key>
    <string>${icon_file}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

    # Create launcher script
    cat > "${macos_dir}/mage-launcher" <<LAUNCHER
#!/usr/bin/env bash
# MAGE.app launcher — delegates to the repository mage.sh
exec "${REPO_DIR}/mage.sh" "\$@"
LAUNCHER
    chmod +x "${macos_dir}/mage-launcher"

    echo ""
    echo "✓ MAGE.app installed to ${app_dir}"
    echo "  You should now see MAGE in Launchpad and Spotlight."
    echo "  To launch from the terminal: ${REPO_DIR}/mage.sh"

    # On macOS, --install always provisions Lemonade.
    # Use --build to compile from source instead of downloading.
    if [[ "${BUILD_LEMONADE}" == "true" ]]; then
        install_build_deps
        build_lemonade
    else
        download_lemonade
    fi
}

do_install() {
    case "${PLATFORM}" in
        Darwin)
            do_install_macos
            ;;
        *)
            do_install_linux
            ;;
    esac
}

do_uninstall_linux() {
    local removed=0
    if [[ -f "${DESKTOP_FILE}" ]]; then
        rm -f "${DESKTOP_FILE}"
        echo "Removed ${DESKTOP_FILE}"
        removed=1
    fi
    if [[ -f "${ICON_FILE}" ]]; then
        rm -f "${ICON_FILE}"
        echo "Removed ${ICON_FILE}"
        removed=1
    fi

    if command -v update-desktop-database &>/dev/null; then
        update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true
    fi

    if [[ ${removed} -eq 0 ]]; then
        echo "Nothing to remove — MAGE is not installed."
    else
        echo "✓ MAGE desktop entry removed."
    fi
}

do_uninstall_macos() {
    if [[ -d "${MACOS_APP}" ]]; then
        rm -rf "${MACOS_APP}"
        echo "✓ Removed ${MACOS_APP}"
    else
        echo "Nothing to remove — MAGE.app is not installed."
    fi
}

do_uninstall() {
    case "${PLATFORM}" in
        Darwin)
            do_uninstall_macos
            ;;
        *)
            do_uninstall_linux
            ;;
    esac
}

do_run() {
    ensure_uv
    sync_deps
    echo "── Launching MAGE… ──"
    cd "${REPO_DIR}"
    exec uv run --package mage-client mage "$@"
}

# ── Main ──────────────────────────────────────────────────────────────────────

ACTION="run"
BUILD_LEMONADE=false

case "${1:-}" in
    --install)
        ACTION="install"
        if [[ "${2:-}" == "--build" ]]; then
            BUILD_LEMONADE=true
        fi
        ;;
    --uninstall)
        ACTION="uninstall"
        ;;
    --build)
        ACTION="build"
        BUILD_LEMONADE=true
        ;;
    --help|-h)
        print_usage
        exit 0
        ;;
esac

case "${ACTION}" in
    install)
        do_install
        ;;
    uninstall)
        do_uninstall
        ;;
    build)
        install_build_deps
        build_lemonade
        ;;
    *)
        do_run "$@"
        ;;
esac

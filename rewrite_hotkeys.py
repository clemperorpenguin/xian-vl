import re

with open('apps/mage-client/src/mage/capture/hotkeys.py', 'r') as f:
    lines = f.readlines()

comments_to_remove = [
    "# Track modifier states per device",
    "# Setup devices",
    "# Handle both \"Double-Tap Shift\" and legacy \"Shift+Space\"",
    "# Start listener threads for current devices",
    "# Provide a basic direct mapping for ABS if needed",
    "# Update modifier tracking on any state change",
    "# Check hotkeys only on key down (1)",
    "# Double-tap detected!",
    "# KEY_ESC is 1",
    "# Check cinematic trigger globally (if mode is active)",
    "# KEY_GRAVE is 41 (backtick/tilde key)",
    "# KEY_C is 46",
    "# KEY_A is 30",
    "# KEY_S is 31",
    "# KEY_M is 50",
    "# KEY_T is 20",
    "# KEY_R is 19",
    "# KEY_H is 35"
]

new_lines = []
for line in lines:
    stripped = line.strip()
    if stripped in comments_to_remove:
        continue
    
    # Also clean up inline comments
    if '  # Shift' in line:
        line = line.replace('  # Shift', '')
    if '  # Ctrl' in line:
        line = line.replace('  # Ctrl', '')
    if '  # Alt' in line:
        line = line.replace('  # Alt', '')

    new_lines.append(line)

with open('apps/mage-client/src/mage/capture/hotkeys.py', 'w') as f:
    f.writelines(new_lines)


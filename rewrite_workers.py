import re

with open('apps/mage-client/src/mage/workers.py', 'r') as f:
    content = f.read()

# Add sanitization function
sanitization_func = '''
def sanitize_markdown(text: str) -> str:
    """Escape backticks and HTML tags to prevent markdown injection."""
    return text.replace("`", "'").replace("<", "&lt;").replace(">", "&gt;")

'''

# Insert the sanitization function after extract_translation_json
target = "class InferenceWorker(QThread):"
content = content.replace(target, sanitization_func + target)

# Replace the write call to use sanitize_markdown
old_write = 'f.write(f"**Source**: {transcript}\\n\\n**Translation**: {translation}\\n\\n---\\n")'
new_write = 'f.write(f"**Source**: {sanitize_markdown(transcript)}\\n\\n**Translation**: {sanitize_markdown(translation)}\\n\\n---\\n")'
content = content.replace(old_write, new_write)

# Remove the redundant comments
comments_to_remove = [
    "    # (list_of_TranslationResult, action_string)\n",
    "    # (partial_translation_text, action_string)\n",
    "    # emitted as soon as run() starts\n",
    "    # (chat_response_text,)\n",
    "    # (accumulated_text,)\n",
    "    # (final_text, still_truncated: bool)\n",
    "    # (original_text, translated_text, audio_bytes)\n",
    "    # (is_available, list_of_model_ids, raw_models_data)\n",
    "    pull_done = pyqtSignal(bool, str)  # (success, message)",
    "    pull_done = pyqtSignal(bool, str)"
]

for c in comments_to_remove:
    content = content.replace(c, "")
    
# Replace the inline pull_done correctly since it was stripped badly above
content = content.replace("    pull_done = pyqtSignal(bool, str)", "    pull_done = pyqtSignal(bool, str)")
content = content.replace("pull_done = pyqtSignal(bool, str)  ", "pull_done = pyqtSignal(bool, str)\n")

with open('apps/mage-client/src/mage/workers.py', 'w') as f:
    f.write(content)


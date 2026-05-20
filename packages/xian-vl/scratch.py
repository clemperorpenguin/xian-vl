import re

text = """**ORIGINAL**
some text here
**TRANSLATED**
some translation here: it is good
**CONFIDENCE**
0.9
"""

orig_match = re.search(
    r'ORIGINAL[a-zA-Z]*[^\w\n]*[:\n][ \t]*(.*?)(?=\n\s*(?:\d+\.\s*)?\**\s*(?:TRANSLAT|CONFIDENCE)|\Z)',
    text, re.DOTALL | re.IGNORECASE
)
trans_match = re.search(
    r'TRANSLAT[a-zA-Z]*[^\w\n]*[:\n][ \t]*(.*?)(?=\n\s*(?:\d+\.\s*)?\**\s*(?:CONFIDENCE|ORIGINAL)|\Z)',
    text, re.DOTALL | re.IGNORECASE
)

print("ORIG:", repr(orig_match.group(1).strip()) if orig_match else None)
print("TRANS:", repr(trans_match.group(1).strip()) if trans_match else None)

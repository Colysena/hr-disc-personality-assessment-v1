import json

raw = """YOUR_RAW_STRING_HERE"""

print("Raw string is:", repr(raw))  # repr() helps show special characters
data = json.loads(raw)
print("Parsed JSON:", data)

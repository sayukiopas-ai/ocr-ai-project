import base64
from openai import OpenAI
from pathlib import Path
from PIL import Image
import io

# Create a dummy image
img = Image.new('RGB', (512, 512), color='white')
buffer = io.BytesIO()
img.save(buffer, format='PNG')
image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
try:
    response = client.chat.completions.create(
        model="typhoon-ocr-small",
        messages=[{"role": "user", "content": [{"type": "text", "text": "อ่านข้อความในภาพ"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}]}]
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)

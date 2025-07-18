import os
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from together import Together
from dotenv import load_dotenv
import json

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
with open("data/menu.json", "r", encoding="utf-8") as f:
    MENU = json.load(f)


client = Together(api_key=TOGETHER_API_KEY)

app = FastAPI()

# CORS 許可（Angularローカル用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では限定すること
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    import json

    # Step 1: 画像を base64 に
    image_bytes = await file.read()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"
    data_uri = f"data:{mime};base64,{base64_str}"

    # Step 2: Together に問い合わせ
    prompt = '''
        You are an expert receipt parser.

        Reply with valid JSON only. DO NOT use Markdown or any explanations. Do not wrap your JSON in triple backticks. Just return pure JSON.
        {
        "items":[{ "name":string, "quantity":number, "price":number }],
        "total":number,
        "service_charge_10_percent":boolean
        }

        Guidelines:
        • "quantity" = number of units (1, 2 …).
        • "price"    = unit price (not subtotal).
        • Ignore tips and discounts.
        • If a 10% service charge is present, set "service_charge_10_percent" to true.
        • Reply with JSON only. Do not add any explanation, description, or Markdown. Just return pure valid JSON.
    '''


    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ],
        max_tokens=1024,
    )

    # Step 3: レスポンス処理 ←⭐ここ！
    import re
    raw_content = response.choices[0].message.content
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
    try:
        if match:
            parsed = json.loads(match.group(1))
        else:
            parsed = json.loads(raw_content)
    except Exception as e:
        return {"error": "invalid JSON", "raw": raw_content}

    # Step 4: 照合・整形
    validated_items = []
    for item in parsed.get("items", []):
        name = item.get("name")
        quantity = item.get("quantity", 1)
        price = item.get("price", 0)
        expected_price = MENU.get(name)
        is_valid = expected_price is not None and abs(expected_price - price) < 0.01

        validated_items.append({
            "name": name,
            "quantity": quantity,
            "price": price,
            "expectedPrice": expected_price,
            "valid": is_valid
        })

    total = sum(
        (item["expectedPrice"] if item["expectedPrice"] is not None else item["price"]) * item["quantity"]
        for item in validated_items
    )

    return {
        "items": validated_items,
        "total": round(total, 2),
        "service_charge_10_percent": parsed.get("service_charge_10_percent", False)
    }

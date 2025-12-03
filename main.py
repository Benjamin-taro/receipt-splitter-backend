import os
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

with open("data/menu.json", "r", encoding="utf-8") as f:
    MENU = json.load(f)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    # Step 1: read image bytes
    image_bytes = await file.read()
    mime = file.content_type or "image/jpeg"

    # Step 2: prompt as before
    prompt = """
        You are an expert receipt parser.

        Reply with valid JSON only. Do NOT use Markdown or explanations.

        Return exactly:
        {
          "items":[{ "name":string, "quantity":number, "price":number }],
          "total":number,
          "service_charge_10_percent":boolean
        }

        Guidelines:
        - "quantity" = number of units.
        - "price" = unit price.
        - Ignore discounts/tips.
        - If a 10% service charge appears, set service_charge_10_percent = true.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")

    # Step 3: Gemini Vision API call
    response = model.generate_content(
        [
            prompt,
            {"mime_type": mime, "data": image_bytes}
        ],
        generation_config={"temperature": 0},  # JSON安定
    )

    raw = response.text.strip()

    # --- remove ```json ... ``` or ``` ... ``` fences if they exist ---
    if raw.startswith("```"):
        # まずバッククォート全部剥がす（```json ... ``` でも ``` ... ``` でも対応）
        raw = raw.strip("`").strip()
        # 先頭が "json\n" / "json\r\n" みたいになってたら取る
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    # Step 4: Parse JSON
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"error": "Invalid JSON", "raw": raw}

    # Step 5: Validate against MENU.json
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
        (i["expectedPrice"] if i["expectedPrice"] is not None else i["price"]) * i["quantity"]
        for i in validated_items
    )

    return {
        "items": validated_items,
        "total": round(total, 2),
        "service_charge_10_percent": parsed.get("service_charge_10_percent", False)
    }

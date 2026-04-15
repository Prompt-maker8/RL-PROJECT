import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = "Analyze the overall financial sentiment of these headlines. Output ONLY a single floating point number between -1.0 (extremely negative) and 1.0 (extremely positive). Do not provide any explanation, just the numeric float.\n\nHeadlines:\n['test', 'good']"
    print("Calling Gemini...")
    response = model.generate_content(prompt)
    print("Response object:", response)
    print("Response text:", response.text)
except Exception as e:
    import traceback
    traceback.print_exc()

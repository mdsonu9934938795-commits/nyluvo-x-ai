from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import os
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client
import time
from datetime import date

load_dotenv()

app = FastAPI(title="NYLUVO X AI Master Engine", version="30.0")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        pass

MASTER_SYSTEM_PROMPT = """You are **NYLUVO X AI**, an advanced multimodal AI assistant developed by **NYLUVO X AI Pvt. Ltd.** and founded by **Mr. Sonu**.

# IDENTITY
- Your name is **NYLUVO X AI**.
- You were founded by **Mr. Sonu** and developed under **NYLUVO X AI Pvt. Ltd.**
- Never claim to be ChatGPT, Gemini, Claude, Copilot, Grok, or any other AI assistant.
- If asked who created or founded you, reply: **"I was founded by Mr. Sonu and developed by NYLUVO X AI Pvt. Ltd."**

# PERSONALITY
You are intelligent, calm, confident, friendly, professional, helpful, honest, respectful, fast, and natural. Your conversation should feel human—not robotic. Automatically detect the user's language and reply in the same language.

# RESPONSE STYLE
Always answer the user's question first, then provide explanation if necessary. Keep responses concise unless more detail is requested. Organize long answers using headings and bullet points.

# KNOWLEDGE & SEARCH RULES
Use your own knowledge first. Do NOT perform web search for programming, coding, debugging, mathematics, physics, chemistry, biology, history, grammar, writing, translation, or general reasoning. Only search if the user explicitly asks for latest, today, current, recent, live, news, weather, stock, crypto, price, election results, sports scores, or if real-time info is needed."""

user_search_counts = {}

def check_and_update_search_quota(user_id: str) -> bool:
    today_str = str(date.today())
    if user_id not in user_search_counts:
        user_search_counts[user_id] = {"date": today_str, "count": 0}
    
    user_data = user_search_counts[user_id]
    if user_data["date"] != today_str:
        user_data["date"] = today_str
        user_data["count"] = 0
        
    if user_data["count"] < 5:
        user_data["count"] += 1
        return True
    return False

async def get_cached_search(query: str) -> str:
    if not supabase:
        return ""
    try:
        res = supabase.table("search_cache").select("result, timestamp").eq("query", query.lower().strip()).execute()
        if res.data:
            row = res.data[0]
            if time.time() - row["timestamp"] < 86400:
                return row["result"]
    except Exception:
        pass
    return ""

async def save_cached_search(query: str, result: str):
    if not supabase:
        return
    try:
        supabase.table("search_cache").upsert({
            "query": query.lower().strip(),
            "result": result,
            "timestamp": int(time.time())
        }).execute()
    except Exception:
        pass

async def tavily_web_search(query: str, user_id: str) -> str:
    if not check_and_update_search_quota(user_id):
        return "" 

    cached = await get_cached_search(query)
    if cached:
        return cached

    tavily_keys = [
        os.getenv("TAVILY_API_KEY_1"), 
        os.getenv("TAVILY_API_KEY_2"),
        os.getenv("TAVILY_API_KEY_3"),
        os.getenv("TAVILY_API_KEY_4")
    ]
    
    async with httpx.AsyncClient(timeout=8.0) as client:
        for key in tavily_keys:
            if not key:
                continue
            try:
                res = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": key, "query": query, "search_depth": "basic", "max_results": 3}
                )
                if res.status_code == 200:
                    data = res.json()
                    results = data.get("results", [])
                    if results:
                        snippets = [r.get("content", "") for r in results]
                        final_text = "[Web Reference Context]: " + " ".join(snippets)
                        await save_cached_search(query, final_text)
                        return final_text
            except Exception:
                continue
    return ""

async def call_ai_with_failover(prompt: str, user_id: str, image_data: str = None) -> str:
    system_prompt = MASTER_SYSTEM_PROMPT
    
    search_triggers = ["latest", "today", "current", "recent", "live", "news", "weather", "stock", "crypto", "price", "election", "score"]
    needs_search = any(w in prompt.lower() for w in search_triggers)
    
    if needs_search:
        web_context = await tavily_web_search(prompt, user_id)
        if web_context:
            system_prompt += f"\n\nReal-time reference data: {web_context}"

    providers = [
        ("Groq-1", "https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_API_KEY_1"), "llama-3.3-70b-versatile", "bearer"),
        ("Groq-2", "https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_API_KEY_2"), "llama-3.3-70b-versatile", "bearer"),
        ("Cerebras-1", "https://api.cerebras.ai/v1/chat/completions", os.getenv("CEREBRAS_API_KEY_1"), "llama3.1-70b", "bearer"),
        ("Cerebras-2", "https://api.cerebras.ai/v1/chat/completions", os.getenv("CEREBRAS_API_KEY_2"), "llama3.1-70b", "bearer"),
        ("Gemini-1", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent", os.getenv("GEMINI_API_KEY_1"), "gemini", "query"),
        ("Gemini-2", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent", os.getenv("GEMINI_API_KEY_2"), "gemini", "query"),
        ("Mistral-1", "https://api.mistral.ai/v1/chat/completions", os.getenv("MISTRAL_API_KEY_1"), "mistral-small-latest", "bearer"),
        ("Mistral-2", "https://api.mistral.ai/v1/chat/completions", os.getenv("MISTRAL_API_KEY_2"), "mistral-small-latest", "bearer"),
        ("Qwen-1", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", os.getenv("QWEN_API_KEY_1"), "qwen-max", "bearer"),
        ("Qwen-2", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", os.getenv("QWEN_API_KEY_2"), "qwen-max", "bearer"),
        ("Cohere-1", "https://api.cohere.com/v1/chat", os.getenv("COHERE_API_KEY_1"), "command-r-plus", "cohere"),
        ("Cohere-2", "https://api.cohere.com/v1/chat", os.getenv("COHERE_API_KEY_2"), "command-r-plus", "cohere")
    ]

    async with httpx.AsyncClient(timeout=35.0) as client:
        for name, url, key, model, auth_type in providers:
            if not key:
                continue
            try:
                if auth_type == "bearer":
                    messages = [{"role": "system", "content": system_prompt}]
                    content = [{"type": "text", "text": prompt if prompt else "Analyze this image."}]
                    if image_data:
                        content.append({"type": "image_url", "image_url": {"url": image_data}})
                    messages.append({"role": "user", "content": content})

                    response = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": messages}
                    )
                    if response.status_code == 200:
                        return response.json()["choices"][0]["message"]["content"]
                        
                elif auth_type == "query":
                    full_p = f"System: {system_prompt}\nUser: {prompt}"
                    response = await client.post(f"{url}?key={key}", json={"contents": [{"parts": [{"text": full_p}]}]})
                    if response.status_code == 200:
                        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                        
                elif auth_type == "cohere":
                    response = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": model, "message": prompt, "preamble": system_prompt}
                    )
                    if response.status_code == 200:
                        return response.json()["text"]
            except Exception:
                continue

    return "All multi-cluster AI nodes are currently busy. Please check your system API keys."

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        image_data = data.get("image", None)
        user_id = data.get("user_id", "default_guest")
        
        if not user_message and not image_data:
            raise HTTPException(status_code=400, detail="Content required")
            
        ai_reply = await call_ai_with_failover(user_message, user_id, image_data)
        return {"response": ai_reply}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

@app.post("/auth/signup")
async def signup(request: Request):
    if not supabase:
        return JSONResponse(status_code=400, content={"error": "Database not configured"})
    data = await request.json()
    try:
        supabase.auth.sign_up({"email": data.get("email"), "password": data.get("password")})
        return {"message": "Account created successfully! Please log in."}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/auth/login")
async def login(request: Request):
    if not supabase:
        return JSONResponse(status_code=400, content={"error": "Database not configured"})
    data = await request.json()
    try:
        res = supabase.auth.sign_in_with_password({"email": data.get("email"), "password": data.get("password")})
        return {"session": res.session.access_token, "user": res.user.email}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid email or password"})

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NYLUVO X Admin - Secure Panel</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #0b0f19; --bg-card: #111827; --border-color: rgba(255, 255, 255, 0.08);
                --text-main: #f9fafb; --text-muted: #9ca3af; --accent: #2563eb; --accent-hover: #1d4ed8;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
            body { background: var(--bg-main); color: var(--text-main); padding: 40px; display: flex; justify-content: center; }
            .admin-container { width: 100%; max-width: 900px; display: flex; flex-direction: column; gap: 24px; }
            .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
            .card { background: var(--bg-card); border: 1px solid var(--border-color); padding: 24px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
            .card h4 { color: var(--text-muted); font-size: 12px; text-transform: uppercase; margin-bottom: 8px; font-weight: 600; letter-spacing: 0.5px; }
            .card .value { font-size: 24px; font-weight: 700; color: #60a5fa; }
            .btn { background: var(--accent); color: #fff; padding: 10px 18px; border-radius: 10px; border: none; font-weight: 600; cursor: pointer; transition: background 0.2s; }
            .btn:hover { background: var(--accent-hover); }
            .login-box { background: var(--bg-card); border: 1px solid var(--border-color); padding: 32px; border-radius: 20px; width: 400px; margin: 100px auto; display: flex; flex-direction: column; gap: 16px; box-shadow: 0 20px 40px rgba(0,0,0,0.6); }
            .login-box input { padding: 14px; border-radius: 10px; border: 1px solid var(--border-color); background: #030712; color: var(--text-main); outline: none; font-size: 14px; }
        </style>
    </head>
    <body>
        <div id="loginScreen" class="login-box">
            <h3>🔒 NYLUVO X Admin Lock</h3>
            <p id="lockStatusMsg" style="font-size: 13px; color: var(--text-muted);">Enter master passcode. 3 attempts remaining.</p>
            <input type="password" id="adminPass" placeholder="Master Password">
            <button class="btn" id="loginBtn" onclick="verifyAdmin()">Access Dashboard</button>
        </div>

        <div id="dashboardContent" class="admin-container" style="display:none;">
            <div class="header">
                <h2>⚡ NYLUVO X Admin Control Panel</h2>
                <button class="btn" style="background:#dc2626;" onclick="location.reload()">Logout</button>
            </div>
            <div class="stats-grid">
                <div class="card">
                    <h4>System Status</h4>
                    <div class="value" style="color: #34d399;">16-API FAILOVER ACTIVE</div>
                </div>
                <div class="card">
                    <h4>Founder Info</h4>
                    <div class="value" style="font-size: 18px; color: #f43f5e;">Mr. Sonu (NYLUVO X AI Pvt. Ltd.)</div>
                </div>
            </div>
        </div>

        <script>
            let adminAttempts = 3;
            function verifyAdmin() {
                const pass = document.getElementById('adminPass').value;
                const ADMIN_SECRET = "nyluvoxadmin2026";
                if(adminAttempts <= 0) return;
                if(pass === ADMIN_SECRET) {
                    document.getElementById('loginScreen').style.display = 'none';
                    document.getElementById('dashboardContent').style.display = 'flex';
                } else {
                    adminAttempts--;
                    if(adminAttempts <= 0) {
                        document.getElementById('adminPass').disabled = true;
                        document.getElementById('loginBtn').disabled = true;
                        document.getElementById('lockStatusMsg').innerText = "Dashboard locked due to 3 failed attempts! 🔒";
                        document.getElementById('lockStatusMsg').style.color = "#f87171";
                    } else {
                        document.getElementById('lockStatusMsg').innerText = `Incorrect password! ${adminAttempts} attempts remaining.`;
                        document.getElementById('lockStatusMsg').style.color = "#fbbf24";
                    }
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
async def home_workspace():
    sup_url = SUPABASE_URL or ""
    sup_key = SUPABASE_KEY or ""
    return f"""
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NYLUVO X AI - Master Workspace</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
        <style>
            :root.dark {{
                --bg-main: #0b0f19; --bg-sidebar: #030712; --bg-chat: #111827;
                --border-color: rgba(255, 255, 255, 0.08); --text-main: #f9fafb; --text-muted: #9ca3af; 
                --accent: #2563eb; --accent-hover: #1d4ed8; --hover-bg: #1f2937;
                --shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            }}
            :root.light {{
                --bg-main: #f3f4f6; --bg-sidebar: #ffffff; --bg-chat: #ffffff;
                --border-color: rgba(0, 0, 0, 0.1); --text-main: #111827; --text-muted: #6b7280; 
                --accent: #2563eb; --accent-hover: #1d4ed8; --hover-bg: #e5e7eb;
                --shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }}
            body {{ background: var(--bg-main); color: var(--text-main); display: flex; height: 100vh; height: 100dvh; overflow: hidden; position: relative; transition: background 0.3s; }}
            
            .sidebar {{ width: 280px; background: var(--bg-sidebar); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding: 16px; height: 100%; z-index: 100; transition: background 0.3s; }}
            .brand {{ font-size: 16px; font-weight: 700; color: var(--text-main); margin-bottom: 20px; display: flex; align-items: center; gap: 8px; padding: 4px 8px; letter-spacing: 0.5px; }}
            .new-chat-btn {{ background: var(--accent); color: #fff; border: none; padding: 12px 16px; border-radius: 12px; font-weight: 600; font-size: 14px; cursor: pointer; text-align: left; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; width: 100%; transition: background 0.2s; }}
            .new-chat-btn:hover {{ background: var(--accent-hover); }}
            .chat-history {{ flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; padding: 0 4px; }}
            .chat-history::-webkit-scrollbar {{ width: 4px; }}
            .chat-history::-webkit-scrollbar-thumb {{ background: rgba(150,150,150,0.2); border-radius: 4px; }}
            .history-item {{ padding: 12px 14px; font-size: 13.5px; color: var(--text-muted); border-radius: 10px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s; user-select: none; }}
            .history-item:hover {{ background: var(--hover-bg); color: var(--text-main); }}
            .history-item .del-btn {{ opacity: 0; transition: opacity 0.2s; background: none; border: none; color: #f87171; cursor: pointer; font-size: 14px; padding: 2px 6px; }}
            .history-item:hover .del-btn {{ opacity: 1; }}

            .sidebar-footer {{ border-top: 1px solid var(--border-color); padding-top: 12px; display: flex; flex-direction: column; gap: 6px; }}
            .footer-btn {{ color: var(--text-muted); font-size: 14px; padding: 12px 14px; border-radius: 10px; display: flex; align-items: center; gap: 12px; background: transparent; border: none; width: 100%; cursor: pointer; text-align: left; transition: all 0.2s; }}
            .footer-btn:hover {{ background: var(--hover-bg); color: var(--text-main); }}

            .main-container {{ flex: 1; display: flex; flex-direction: column; background: var(--bg-main); position: relative; height: 100%; min-width: 0; }}
            .chat-header {{ padding: 16px 24px; border-bottom: 1px solid var(--border-color); font-weight: 600; font-size: 14px; display: flex; align-items: center; justify-content: space-between; background: var(--bg-main); z-index: 10; }}
            .chat-messages {{ flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 28px; align-items: center; scroll-behavior: smooth; }}
            .chat-messages::-webkit-scrollbar {{ width: 6px; }}
            .chat-messages::-webkit-scrollbar-thumb {{ background: rgba(150,150,150,0.2); border-radius: 4px; }}
            
            .message-wrapper {{ width: 100%; max-width: 780px; display: flex; gap: 16px; font-size: 15px; line-height: 1.7; position: relative; animation: fadeIn 0.3s ease; }}
            @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            .message-wrapper.user {{ justify-content: flex-end; }}
            .message-bubble {{ padding: 14px 18px; border-radius: 18px; max-width: 82%; word-break: break-word; box-shadow: var(--shadow); }}
            .message-wrapper.user .message-bubble {{ background: var(--accent); color: #fff; border-top-right-radius: 4px; }}
            .message-wrapper.ai .message-bubble {{ background: var(--bg-chat); border: 1px solid var(--border-color); color: var(--text-main); border-top-left-radius: 4px; }}
            
            .typing-cursor::after {{ content: '▋'; display: inline-block; animation: blink 1s infinite; color: var(--accent); margin-left: 2px; font-size: 12px; vertical-align: baseline; }}
            @keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}

            .input-container {{ padding: 16px 24px 28px 24px; background: var(--bg-main); display: flex; justify-content: center; }}
            .input-box {{ width: 100%; max-width: 780px; background: var(--bg-chat); border: 1px solid var(--border-color); border-radius: 24px; display: flex; flex-direction: column; padding: 12px 16px; box-shadow: var(--shadow); transition: border-color 0.2s; }}
            .input-box:focus-within {{ border-color: var(--accent); }}
            .input-top {{ display: flex; align-items: flex-end; gap: 12px; }}
            .input-box textarea {{ flex: 1; background: transparent; border: none; color: var(--text-main); font-size: 15px; resize: none; outline: none; padding: 6px; max-height: 180px; }}
            .input-actions {{ display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }}
            
            .action-tools {{ display: flex; gap: 10px; align-items: center; }}
            .icon-btn {{ background: transparent; border: none; cursor: pointer; color: var(--text-muted); font-size: 18px; display: flex; align-items: center; justify-content: center; transition: color 0.2s; }}
            .icon-btn:hover {{ color: var(--accent); }}
            
            .send-btn {{ background: var(--accent); color: #fff; border: none; width: 38px; height: 38px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; font-weight: bold; transition: background 0.2s, transform 0.1s; }}
            .send-btn:hover {{ background: var(--accent-hover); transform: scale(1.05); }}

            .modal-overlay {{ position: fixed; inset: 0; background: rgba(0,0,0,0.7); backdrop-filter: blur(5px); display: flex; align-items: center; justify-content: center; z-index: 1000; }}
            .modal-card {{ background: var(--bg-chat); border: 1px solid var(--border-color); padding: 32px; border-radius: 24px; width: 420px; display: flex; flex-direction: column; gap: 16px; box-shadow: 0 25px 50px rgba(0,0,0,0.6); }}
            .modal-card input {{ width: 100%; padding: 14px; border-radius: 12px; border: 1px solid var(--border-color); background: var(--bg-main); color: var(--text-main); outline: none; font-size: 14px; }}
            .primary-btn {{ background: var(--accent); color: #fff; padding: 14px; border-radius: 12px; border: none; font-weight: 600; cursor: pointer; font-size: 14px; transition: background 0.2s; }}
            .primary-btn:hover {{ background: var(--accent-hover); }}
            
            .google-btn {{ background: #fff; color: #374151; padding: 12px; border-radius: 12px; border: 1px solid var(--border-color); font-weight: 600; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center; gap: 10px; transition: background 0.2s; }}
            .google-btn:hover {{ background: #f3f4f6; }}
            .divider {{ display: flex; align-items: center; text-align: center; color: var(--text-muted); font-size: 12px; margin: 4px 0; }}
            .divider::before, .divider::after {{ content: ''; flex: 1; border-bottom: 1px solid var(--border-color); }}
            .divider::before {{ margin-right: .75em; }}
            .divider::after {{ margin-left: .75em; }}

            #imagePreviewContainer {{ display: none; padding: 8px 12px; gap: 8px; align-items: center; border-bottom: 1px solid var(--border-color); }}
            #imagePreviewContainer img {{ width: 40px; height: 40px; border-radius: 8px; object-fit: cover; }}
        </style>
    </head>
    <body>
        <div id="authModal" class="modal-overlay" style="display:none;">
            <div class="modal-card">
                <h3 id="authTitle" style="font-size: 18px; font-weight: 700;">🔐 Login to NYLUVO X AI</h3>
                <div id="authError" style="color: #f87171; font-size: 13px; display:none;"></div>
                
                <button class="google-btn" onclick="handleGoogleLogin()">
                    <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.66-5.17 3.66-9.17z"/><path fill="#34A853" d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.13 0-5.78-2.11-6.73-4.96H1.18v3.15C3.15 21.32 7.22 24 12 24z"/><path fill="#FBBC05" d="M5.27 14.24c-.25-.72-.38-1.49-.38-2.24s.13-1.52.38-2.24V6.6H1.18C.43 8.12 0 9.81 0 12s.43 3.88 1.18 5.4l4.09-3.16z"/><path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.22 0 3.15 2.68 1.18 6.6l4.09 3.15c.95-2.85 3.6-4.99 6.73-4.99z"/></svg>
                    Continue with Google
                </button>
                
                <div class="divider">OR</div>

                <input type="email" id="authEmail" placeholder="Email address">
                <input type="password" id="authPassword" placeholder="Password">
                <button class="primary-btn" id="authSubmitBtn" onclick="handleAuthSubmit()">Login</button>
                <div style="display: flex; justify-content: space-between; font-size: 13.5px; color: var(--text-muted); cursor: pointer; margin-top: 4px;">
                    <span id="authToggleText" onclick="toggleAuthMode()">Create an account</span>
                    <span onclick="document.getElementById('authModal').style.display='none'">Cancel</span>
                </div>
            </div>
        </div>

        <div id="settingsModal" class="modal-overlay" style="display:none;">
            <div class="modal-card">
                <h3 style="font-size: 18px; font-weight: 700;">⚙️ Workspace Settings</h3>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                    <span>Appearance Theme</span>
                    <button class="primary-btn" style="padding: 8px 14px;" onclick="toggleTheme()">Switch Dark/Light</button>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                    <span>Clear Local History</span>
                    <button class="primary-btn" style="padding: 8px 14px; background:#dc2626;" onclick="clearHistory()">Clear All</button>
                </div>
                <button class="primary-btn" style="margin-top: 20px;" onclick="document.getElementById('settingsModal').style.display='none'">Close</button>
            </div>
        </div>

        <div class="sidebar" id="appSidebar">
            <div class="brand">
                <span>⚡ NYLUVO X AI</span>
            </div>
            <button class="new-chat-btn" onclick="createNewChat()"><span>New chat</span> <span>＋</span></button>
            
            <div class="chat-history" id="chatHistoryList">
                <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); padding: 4px 6px; font-weight: 700; letter-spacing: 0.5px;">Recent Chats</div>
            </div>

            <div class="sidebar-footer">
                <button class="footer-btn" onclick="document.getElementById('settingsModal').style.display='flex'">⚙️ Settings</button>
                <button class="footer-btn" id="authNavBtn" onclick="openAuthModal('login')">👤 Account Login</button>
            </div>
        </div>

        <div class="main-container">
            <div class="chat-header">
                <span id="currentChatTitle">New Workspace</span>
                <span id="userLoggedInBadge" style="font-size: 12px; color: var(--text-muted);"></span>
            </div>
            
            <div class="chat-messages" id="chatWindow">
                <div class="message-wrapper ai">
                    <div style="width: 32px; height: 32px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 12px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(37,99,235,0.3);">AI</div>
                    <div class="message-bubble">Hello! I was founded by Mr. Sonu and developed by NYLUVO X AI Pvt. Ltd. How can I help you today?</div>
                </div>
            </div>

            <div class="input-container">
                <div class="input-box">
                    <div id="imagePreviewContainer">
                        <img id="previewImg" src="" alt="preview">
                        <span id="previewName" style="font-size: 12px; color: var(--text-muted); flex: 1;"></span>
                        <span onclick="removeImage()" style="cursor: pointer; font-weight: bold; color: #f87171;">✕</span>
                    </div>
                    <div class="input-top">
                        <textarea rows="1" placeholder="Message NYLUVO X AI..." id="userInput"></textarea>
                    </div>
                    <div class="input-actions">
                        <div class="action-tools">
                            <label class="icon-btn" title="Upload Image" style="cursor: pointer;">
                                📎
                                <input type="file" id="imageInput" accept="image/*" style="display:none;" onchange="handleImageSelect(event)">
                            </label>
                        </div>
                        <button class="send-btn" onclick="sendMessage()">↑</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentTheme = localStorage.getItem('nyluvo_theme') || 'dark';
            document.documentElement.className = currentTheme;

            function toggleTheme() {
                currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
                document.documentElement.className = currentTheme;
                localStorage.setItem('nyluvo_theme', currentTheme);
            }

            let uploadedBase64Image = null;
            function handleImageSelect(event) {
                const file = event.target.files[0];
                if(file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        uploadedBase64Image = e.target.result;
                        document.getElementById('previewImg').src = uploadedBase64Image;
                        document.getElementById('previewName').innerText = file.name;
                        document.getElementById('imagePreviewContainer').style.display = 'flex';
                    };
                    reader.readAsDataURL(file);
                }
            }

            function removeImage() {
                uploadedBase64Image = null;
                document.getElementById('imageInput').value = '';
                document.getElementById('imagePreviewContainer').style.display = 'none';
            }

            const SUPABASE_URL = "{sup_url}";
            const SUPABASE_KEY = "{sup_key}";
            let supabaseClient = null;
            if(SUPABASE_URL && SUPABASE_KEY) {
                supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);
            }

            let chats = JSON.parse(localStorage.getItem('chats')) || [{ id: Date.now(), title: 'New Workspace', messages: [] }];
            let activeChatId = chats[0].id;
            let currentUser = localStorage.getItem('nyluvo_user') || null;
            let currentUserId = localStorage.getItem('nyluvo_user_id') || 'user_' + Math.random().toString(36).substring(7);
            localStorage.setItem('nyluvo_user_id', currentUserId);

            if(currentUser) {
                document.getElementById('userLoggedInBadge').innerText = currentUser;
                document.getElementById('authNavBtn').innerText = '🚪 Logout';
            }

            if(supabaseClient) {
                supabaseClient.auth.getSession().then(({ data: { session } }) => {
                    if (session && session.user) {
                        currentUser = session.user.email;
                        localStorage.setItem('nyluvo_user', currentUser);
                        document.getElementById('userLoggedInBadge').innerText = currentUser;
                        document.getElementById('authNavBtn').innerText = '🚪 Logout';
                    }
                });
            }

            async function handleGoogleLogin() {
                if(!supabaseClient) { alert('Supabase credentials missing.'); return; }
                await supabaseClient.auth.signInWithOAuth({ provider: 'google', options: { redirectTo: window.location.origin } });
            }

            let isSignUpMode = false;
            function openAuthModal(mode) {
                if(currentUser) {
                    if(supabaseClient) supabaseClient.auth.signOut();
                    localStorage.removeItem('nyluvo_user');
                    currentUser = null;
                    document.getElementById('userLoggedInBadge').innerText = '';
                    document.getElementById('authNavBtn').innerText = '👤 Account Login';
                    alert('Logged out successfully.');
                    return;
                }
                isSignUpMode = (mode === 'signup');
                document.getElementById('authModal').style.display = 'flex';
            }

            function toggleAuthMode() {
                isSignUpMode = !isSignUpMode;
                document.getElementById('authTitle').innerText = isSignUpMode ? '📝 Create Account' : '🔐 Login to NYLUVO X AI';
                document.getElementById('authSubmitBtn').innerText = isSignUpMode ? 'Sign Up' : 'Login';
                document.getElementById('authToggleText').innerText = isSignUpMode ? 'Already have an account? Login' : 'Create an account';
            }

            async function handleAuthSubmit() {
                const email = document.getElementById('authEmail').value.trim();
                const password = document.getElementById('authPassword').value.trim();
                const errBox = document.getElementById('authError');
                if(!email || !password) { errBox.innerText = 'Please fill all fields'; errBox.style.display = 'block'; return; }

                const endpoint = isSignUpMode ? '/auth/signup' : '/auth/login';
                try {
                    const res = await fetch(endpoint, {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email, password })
                    });
                    const data = await res.json();
                    if(res.ok) {
                        if(isSignUpMode) { alert(data.message); toggleAuthMode(); }
                        else {
                            currentUser = data.user;
                            localStorage.setItem('nyluvo_user', currentUser);
                            document.getElementById('userLoggedInBadge').innerText = currentUser;
                            document.getElementById('authNavBtn').innerText = '🚪 Logout';
                            document.getElementById('authModal').style.display = 'none';
                        }
                    } else {
                        errBox.innerText = data.error || 'Authentication failed';
                        errBox.style.display = 'block';
                    }
                } catch(e) { errBox.innerText = 'Network error occurred'; errBox.style.display = 'block'; }
            }

            function saveChats() { localStorage.setItem('chats', JSON.stringify(chats)); renderHistory(); }
            
            function renderHistory() {
                const list = document.getElementById('chatHistoryList');
                list.innerHTML = '<div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); padding: 4px 6px; font-weight: 700; letter-spacing: 0.5px;">Recent Chats</div>';
                
                chats.forEach(chat => {
                    const item = document.createElement('div');
                    item.className = 'history-item';
                    item.innerHTML = `<span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" onclick="switchChat(${chat.id})">${chat.title}</span><button class="del-btn" onclick="deleteChat(event, ${chat.id})" title="Delete Chat">✕</button>`;
                    
                    // Long press support for mobile devices
                    let pressTimer;
                    item.addEventListener('touchstart', function() {
                        pressTimer = setTimeout(function() {
                            if(confirm("Delete this chat session?")) {
                                deleteChatDirect(${chat.id});
                            }
                        }, 800); // 800ms long press
                    });
                    item.addEventListener('touchend', function() { clearTimeout(pressTimer); });

                    list.appendChild(item);
                });
            }

            function deleteChat(event, id) {
                event.stopPropagation();
                if(chats.length <= 1) {
                    alert("At least one active chat session is required.");
                    return;
                }
                chats = chats.filter(c => c.id !== id);
                if(activeChatId === id) {
                    activeChatId = chats[0].id;
                }
                saveChats();
                loadActiveChat();
            }

            function deleteChatDirect(id) {
                if(chats.length <= 1) return;
                chats = chats.filter(c => c.id !== id);
                if(activeChatId === id) {
                    activeChatId = chats[0].id;
                }
                saveChats();
                loadActiveChat();
            }

            function createNewChat() {
                const newChat = { id: Date.now(), title: 'New Workspace', messages: [] };
                chats.unshift(newChat); activeChatId = newChat.id; saveChats(); loadActiveChat();
            }
            function clearHistory() {
                localStorage.removeItem('chats');
                chats = [{ id: Date.now(), title: 'New Workspace', messages: [] }];
                activeChatId = chats[0].id;
                saveChats(); loadActiveChat();
                document.getElementById('settingsModal').style.display = 'none';
            }
            function switchChat(id) { activeChatId = id; loadActiveChat(); }
            function loadActiveChat() {
                const chat = chats.find(c => c.id === activeChatId);
                if (!chat) return;
                document.getElementById('currentChatTitle').innerText = chat.title;
                const window = document.getElementById('chatWindow');
                window.innerHTML = '';
                if(chat.messages.length === 0) {
                    window.innerHTML = `<div class="message-wrapper ai"><div style="width: 32px; height: 32px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 12px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(37,99,235,0.3);">AI</div><div class="message-bubble">Hello! I was founded by Mr. Sonu and developed by NYLUVO X AI Pvt. Ltd. How can I help you today?</div></div>`;
                } else {
                    chat.messages.forEach(m => {
                        window.innerHTML += `<div class="message-wrapper ${m.role}">${m.role === 'ai' ? '<div style="width: 32px; height: 32px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 12px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(37,99,235,0.3);">AI</div>' : ''}<div class="message-bubble">${m.content}</div></div>`;
                    });
                }
                window.scrollTop = window.scrollHeight;
            }

            const textarea = document.getElementById('userInput');
            textarea.addEventListener('keydown', function(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

            async function typeWriterEffect(bubbleElement, text, speed = 5) {
                bubbleElement.classList.add('typing-cursor');
                let i = 0;
                return new Promise(resolve => {
                    function type() {
                        if (i < text.length) {
                            bubbleElement.innerHTML += text.charAt(i);
                            i++;
                            setTimeout(type, speed);
                        } else {
                            bubbleElement.classList.remove('typing-cursor');
                            resolve();
                        }
                    }
                    type();
                });
            }

            async function sendMessage() {
                const text = textarea.value.trim();
                const currentImg = uploadedBase64Image;
                if (!text && !currentImg) return;

                let chat = chats.find(c => c.id === activeChatId);
                if(chat.messages.length === 0) chat.title = text.length > 25 ? text.substring(0, 25) + '...' : 'New Chat';

                let displayContent = text;
                if(currentImg) displayContent += `<br><img src="${currentImg}" style="max-width:150px; border-radius:8px; margin-top:6px;">`;

                chat.messages.push({ role: 'user', content: displayContent });
                saveChats(); loadActiveChat();
                textarea.value = '';
                removeImage();

                const chatWindow = document.getElementById('chatWindow');
                const aiWrapper = document.createElement('div');
                aiWrapper.className = 'message-wrapper ai';
                aiWrapper.innerHTML = `<div style="width: 32px; height: 32px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 12px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(37,99,235,0.3);">AI</div><div class="message-bubble"></div>`;
                chatWindow.appendChild(aiWrapper);
                chatWindow.scrollTop = chatWindow.scrollHeight;
                const bubble = aiWrapper.querySelector('.message-bubble');

                try {
                    const response = await fetch('/chat', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text, image: currentImg, user_id: currentUserId })
                    });
                    const data = await response.json();
                    
                    await typeWriterEffect(bubble, data.response);

                    chat.messages.push({ role: 'ai', content: data.response });
                    saveChats();
                } catch (err) {
                    bubble.style.color = "#f87171";
                    bubble.innerText = "Connection error. Please try again.";
                }
            }
            renderHistory(); loadActiveChat();
        </script>
    </body>
    </html>
    """

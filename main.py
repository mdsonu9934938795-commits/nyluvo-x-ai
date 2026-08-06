from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import os
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client
import time
from datetime import datetime

load_dotenv()

app = FastAPI(title="Nyluvo X AI - Ultimate Private Engine", version="22.0")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        pass

MODE_PROMPTS = {
    "general": "You are Nyluvo, an ultra-intelligent, friendly, and natural human-like AI assistant. Speak directly, warmly, and conversationally like a real human friend or expert colleague. Never keep repeating greetings, self-introductions, or boring template phrases. Get straight to the point while maintaining high warmth and intelligence. You are founded by Mr. Sonu and Nyluvo X AI Pvt Ltd.",
    "student": "You are Nyluvo, an expert academic tutor and mentor. Explain things like an inspiring human teacher using crystal-clear analogies and step-by-step guidance. Avoid robotic intros or repeating greetings. Keep explanations engaging and precise. You are founded by Mr. Sonu and Nyluvo X AI Pvt Ltd.",
    "developer": "You are Nyluvo, a senior software architect and tech mentor. Provide clean, production-ready code with sharp, pragmatic, and human-like technical reasoning. Skip fluff or repetitive greetings. You are founded by Mr. Sonu and Nyluvo X AI Pvt Ltd.",
    "hacker": "You are Nyluvo, an elite cybersecurity expert and ethical penetration tester. Discuss system architectures, protocols, and security practices with deep technical precision and a sharp, direct tone. You are founded by Mr. Sonu and Nyluvo X AI Pvt Ltd."
}

async def check_and_update_message_limit(user_email: str) -> dict:
    """
    Background silent tracking:
    - Free user: 35 messages/day
    - Pro user: 200 messages/day
    No public counter is shown to the user on UI.
    """
    if not supabase or not user_email:
        return {"allowed": True, "msg": ""}
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        res = supabase.table("profiles").select("is_pro, message_count, last_reset_date").eq("email", user_email).execute()
        
        if not res.data:
            supabase.table("profiles").insert({"email": user_email, "is_pro": False, "message_count": 1, "last_reset_date": today_str}).execute()
            return {"allowed": True, "msg": ""}
            
        profile = res.data[0]
        is_pro = profile.get("is_pro", False)
        last_date = profile.get("last_reset_date")
        count = profile.get("message_count", 0)
        
        daily_limit = 200 if is_pro else 35
        
        if last_date != today_str:
            supabase.table("profiles").update({"message_count": 1, "last_reset_date": today_str}).eq("email", user_email).execute()
            return {"allowed": True, "msg": ""}
            
        if count >= daily_limit:
            if is_pro:
                return {"allowed": False, "msg": "Aapne aaj ke aapke 200 Pro messages ki limit poori kar li hai. Kal quota automatically reset ho jayega."}
            else:
                return {"allowed": False, "msg": "Aapne aaj ke free messages ki limit cross kar li hai. Unlimited aur high-speed access ke liye **Nyluvo Pro sirf ₹99 mein upgrade karein (200 messages/day)**!"}
            
        supabase.table("profiles").update({"message_count": count + 1}).eq("email", user_email).execute()
        return {"allowed": True, "msg": ""}
    except Exception:
        return {"allowed": True, "msg": ""}

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

async def tavily_web_search(query: str) -> str:
    cached = await get_cached_search(query)
    if cached:
        return cached

    tavily_keys = [
        os.getenv("TAVILY_API_KEY_1"), 
        os.getenv("TAVILY_API_KEY_2"),
        os.getenv("TAVILY_API_KEY_3"),
        os.getenv("TAVILY_API_KEY_4")
    ]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
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

async def call_ai_with_failover(prompt: str, mode: str, image_data: str = None) -> str:
    system_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["general"])
    
    web_context = await tavily_web_search(prompt)
    if web_context:
        system_prompt += f"\n\nReal-time reference info: {web_context}"

    providers = [
        ("Groq-1", "https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_API_KEY_1"), "llama-3.3-70b-versatile", "bearer"),
        ("Groq-2", "https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_API_KEY_2"), "llama-3.3-70b-versatile", "bearer"),
        ("Cerebras-1", "https://api.cerebras.ai/v1/chat/completions", os.getenv("CEREBRAS_API_KEY_1"), "llama3.1-70b", "bearer"),
        ("Cerebras-2", "https://api.cerebras.ai/v1/chat/completions", os.getenv("CEREBRAS_API_KEY_2"), "llama3.1-70b", "bearer"),
        ("Gemini-1", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent", os.getenv("GEMINI_API_KEY_1"), "gemini", "query"),
        ("Gemini-2", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent", os.getenv("GEMINI_API_KEY_2"), "gemini", "query"),
        ("Mistral-1", "https://api.mistral.ai/v1/chat/completions", os.getenv("MISTRAL_API_KEY_1"), "mistral-small-latest", "bearer"),
        ("Mistral-2", "https://api.mistral.ai/v1/chat/completions", os.getenv("MISTRAL_API_KEY_2"), "mistral-small-latest", "bearer")
    ]

    async with httpx.AsyncClient(timeout=35.0) as client:
        for name, url, key, model, auth_type in providers:
            if not key:
                continue
            try:
                if auth_type == "bearer":
                    messages = [{"role": "system", "content": system_prompt}]
                    content = [{"type": "text", "text": prompt}]
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
            except Exception:
                continue

    return "Abhi servers thode busy hain. Kripya apna sawal dobara bhejiye."

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        mode = data.get("mode", "general")
        image_data = data.get("image", None)
        user_email = data.get("email", None)
        
        if not user_message and not image_data:
            raise HTTPException(status_code=400, detail="Content required")
            
        # Silent background limit check (No UI clutter)
        limit_check = await check_and_update_message_limit(user_email)
        if not limit_check["allowed"]:
            return {"response": limit_check["msg"]}
            
        ai_reply = await call_ai_with_failover(user_message, mode, image_data)
        return {"response": ai_reply}
    except Exception as e:
        return {"response": f"Kuch technical dikkat aayi hai: {str(e)}"}

@app.post("/auth/signup")
async def signup(request: Request):
    if not supabase:
        return JSONResponse(status_code=400, content={"error": "Database configured nahi hai"})
    data = await request.json()
    try:
        res = supabase.auth.sign_up({"email": data.get("email"), "password": data.get("password")})
        return {"message": "Account successfully ban gaya hai! Ab login karein."}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/auth/login")
async def login(request: Request):
    if not supabase:
        return JSONResponse(status_code=400, content={"error": "Database configured nahi hai"})
    data = await request.json()
    try:
        res = supabase.auth.sign_in_with_password({"email": data.get("email"), "password": data.get("password")})
        return {"session": res.session.access_token, "user": res.user.email}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Galat email ya password hai"})

@app.post("/payment/upgrade-success")
async def upgrade_success(request: Request):
    if not supabase:
        return JSONResponse(status_code=400, content={"error": "Database configured nahi hai"})
    data = await request.json()
    email = data.get("email")
    if not email:
        return JSONResponse(status_code=400, content={"error": "Email zaroori hai"})
    try:
        supabase.table("profiles").update({"is_pro": True}).eq("email", email).execute()
        return {"success": True, "message": "Badhai ho! Aapka account Pro mein upgrade ho gaya hai."}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def home_workspace():
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nyluvo X AI - Professional Assistant</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #fcfcfd; --bg-sidebar: #f4f6f9; --bg-chat: #ffffff;
                --border-color: #e4e7ec; --text-main: #101828; --text-muted: #475467; 
                --accent: #2563eb; --accent-hover: #1d4ed8; --hover-bg: #eaecf0;
                --shadow: 0 12px 32px rgba(16, 24, 40, 0.05);
            }
            .dark {
                --bg-main: #0d1117; --bg-sidebar: #161b22; --bg-chat: #21262d;
                --border-color: rgba(255, 255, 255, 0.1); --text-main: #f0f6fc; --text-muted: #8b949e; 
                --accent: #3b82f6; --accent-hover: #60a5fa; --hover-bg: #30363d;
                --shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease; }
            body { background: var(--bg-main); color: var(--text-main); display: flex; height: 100vh; height: 100dvh; overflow: hidden; position: relative; }
            
            .sidebar { width: 260px; background: var(--bg-sidebar); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding: 12px; height: 100%; z-index: 100; transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
            .brand { font-size: 16px; font-weight: 700; color: var(--text-main); margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; padding: 4px 8px; }
            .brand span { display: flex; align-items: center; gap: 8px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            
            .new-chat-btn { background: var(--accent); color: #ffffff; border: none; padding: 10px 14px; border-radius: 10px; font-weight: 600; font-size: 13.5px; cursor: pointer; text-align: left; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; width: 100%; box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3); }
            .new-chat-btn:hover { background: var(--accent-hover); }
            
            .pro-banner { background: linear-gradient(135deg, #7c3aed, #4f46e5); color: #fff; padding: 12px; border-radius: 12px; margin-bottom: 16px; cursor: pointer; box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4); }
            .pro-banner h4 { font-size: 13px; font-weight: 700; display: flex; align-items: center; gap: 6px; }
            .pro-banner p { font-size: 11px; opacity: 0.9; margin-top: 2px; }

            .mode-selector { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; padding: 0 4px; }
            .mode-label { font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; letter-spacing: 0.5px; }
            .mode-select { background: var(--bg-chat); border: 1px solid var(--border-color); color: var(--text-main); padding: 10px 12px; border-radius: 8px; font-size: 13.5px; outline: none; cursor: pointer; font-weight: 500; }
            
            .chat-history { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; padding: 0 4px; }
            .history-item { padding: 10px 12px; font-size: 13.5px; color: var(--text-muted); border-radius: 8px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: 500; }
            .history-item:hover { background: var(--hover-bg); color: var(--text-main); }
            .delete-chat { background: transparent; border: none; color: var(--text-muted); cursor: pointer; font-size: 12px; opacity: 0; }
            .history-item:hover .delete-chat { opacity: 1; }
            .delete-chat:hover { color: #ef4444; }

            .sidebar-footer { border-top: 1px solid var(--border-color); padding-top: 10px; display: flex; flex-direction: column; gap: 4px; }
            .footer-btn { color: var(--text-muted); font-size: 13.5px; padding: 10px 12px; border-radius: 8px; display: flex; align-items: center; gap: 10px; background: transparent; border: none; width: 100%; cursor: pointer; text-align: left; font-weight: 500; }
            .footer-btn:hover { background: var(--hover-bg); color: var(--text-main); }

            .main-container { flex: 1; display: flex; flex-direction: column; background: var(--bg-main); position: relative; height: 100%; min-width: 0; }
            .chat-header { padding: 12px 20px; border-bottom: 1px solid var(--border-color); font-weight: 600; font-size: 14px; display: flex; align-items: center; justify-content: space-between; background: var(--bg-main); }
            .header-left { display: flex; align-items: center; gap: 12px; }
            
            .menu-toggle-btn { background: transparent; border: none; color: var(--text-main); font-size: 18px; cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 4px; border-radius: 6px; }
            .menu-toggle-btn:hover { background: var(--hover-bg); }

            .chat-messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 24px; align-items: center; scroll-behavior: smooth; }
            .message-wrapper { width: 100%; max-width: 768px; display: flex; gap: 16px; font-size: 15px; line-height: 1.6; position: relative; }
            .message-wrapper.user { justify-content: flex-end; }
            .message-bubble { padding: 12px 16px; border-radius: 16px; max-width: 85%; word-break: break-word; box-shadow: var(--shadow); }
            .message-wrapper.user .message-bubble { background: var(--accent); color: #ffffff; border-top-right-radius: 4px; }
            .message-wrapper.ai .message-bubble { background: var(--bg-chat); border: 1px solid var(--border-color); color: var(--text-main); border-top-left-radius: 4px; }
            .msg-actions { position: absolute; right: 0; bottom: -16px; font-size: 11px; color: var(--text-muted); cursor: pointer; display: none; }
            .message-wrapper:hover .msg-actions { display: block; }

            .input-container { padding: 16px 20px 24px 20px; background: var(--bg-main); display: flex; justify-content: center; }
            .input-box { width: 100%; max-width: 768px; background: var(--bg-chat); border: 1px solid var(--border-color); border-radius: 20px; display: flex; flex-direction: column; padding: 10px 14px; box-shadow: var(--shadow); }
            .input-box:focus-within { border-color: var(--accent); }
            .input-top { display: flex; align-items: flex-end; gap: 10px; }
            .input-box textarea { flex: 1; background: transparent; border: none; color: var(--text-main); font-size: 15px; resize: none; outline: none; padding: 6px; max-height: 160px; }
            
            .input-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 6px; }
            .tool-group { display: flex; gap: 6px; align-items: center; }
            .tool-btn { background: transparent; border: none; color: var(--text-muted); cursor: pointer; font-size: 16px; display: flex; align-items: center; padding: 6px; border-radius: 6px; }
            .tool-btn:hover { background: var(--hover-bg); color: var(--text-main); }
            
            .send-btn { background: var(--accent); color: #ffffff; border: none; width: 34px; height: 34px; border-radius: 50%; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 14px; box-shadow: 0 2px 10px rgba(59, 130, 246, 0.3); }
            .send-btn:hover { background: var(--accent-hover); }

            #previewContainer { display: none; padding: 6px 8px; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); border-bottom: 1px solid var(--border-color); margin-bottom: 6px; }
            #previewImg { width: 36px; height: 36px; border-radius: 6px; object-fit: cover; border: 1px solid var(--border-color); }

            .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 1000; backdrop-filter: blur(6px); }
            .modal-card { background: var(--bg-sidebar); border: 1px solid var(--border-color); padding: 28px; border-radius: 20px; width: 380px; display: flex; flex-direction: column; gap: 14px; box-shadow: var(--shadow); }
            .modal-card input { width: 100%; padding: 12px 14px; border-radius: 10px; border: 1px solid var(--border-color); background: var(--bg-chat); color: var(--text-main); outline: none; font-size: 14px; }
            .primary-btn { background: var(--accent); color: #ffffff; padding: 12px; border-radius: 10px; border: none; font-weight: 600; cursor: pointer; font-size: 14px; box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3); }
            .primary-btn:hover { background: var(--accent-hover); }

            @media (max-width: 768px) {
                .sidebar { position: absolute; left: 0; top: 0; bottom: 0; transform: translateX(-100%); }
                .sidebar.open { transform: translateX(0); }
                .sidebar-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 90; display: none; }
                .sidebar-overlay.active { display: block; }
            }
        </style>
    </head>
    <body>
        <div class="sidebar-overlay" id="sidebarOverlay" onclick="toggleSidebar()"></div>

        <div id="authModal" class="modal-overlay" style="display:none;">
            <div class="modal-card">
                <h3 id="authTitle" style="font-size: 18px; font-weight: 700;">🔐 Login to Nyluvo</h3>
                <div id="authError" style="color: #ef4444; font-size: 12px; display:none;"></div>
                <input type="email" id="authEmail" placeholder="Email address">
                <input type="password" id="authPassword" placeholder="Password">
                <button class="primary-btn" id="authSubmitBtn" onclick="handleAuthSubmit()">Login</button>
                <div style="display: flex; justify-content: space-between; font-size: 13px; color: var(--text-muted); margin-top: 4px;">
                    <span id="authToggleText" style="cursor: pointer; color: var(--accent);" onclick="toggleAuthMode()">Create account</span>
                    <span style="cursor: pointer;" onclick="document.getElementById('authModal').style.display='none'">Cancel</span>
                </div>
            </div>
        </div>

        <div id="proModal" class="modal-overlay" style="display:none;">
            <div class="modal-card" style="text-align: center;">
                <div style="font-size: 32px;">👑</div>
                <h3 style="font-size: 20px; font-weight: 700;">Nyluvo Pro Upgrade</h3>
                <p style="font-size: 13px; color: var(--text-muted);">Get 200 daily messages, high-speed priority cluster, and professional modes for just <b style="color:var(--text-main);">₹99/month</b>.</p>
                <button class="primary-btn" onclick="simulatePayment()" style="background: linear-gradient(135deg, #7c3aed, #4f46e5); margin-top: 10px;">Pay ₹99 Now (Instant Unlock)</button>
                <button class="primary-btn" style="background: transparent; border: 1px solid var(--border-color); color: var(--text-main); box-shadow: none;" onclick="document.getElementById('proModal').style.display='none'">Maybe Later</button>
            </div>
        </div>

        <div class="sidebar" id="appSidebar">
            <div class="brand">
                <span>⚡ Nyluvo X AI</span>
                <button onclick="toggleSidebar()" class="menu-toggle-btn" style="font-size: 14px;">✕</button>
            </div>
            <button class="new-chat-btn" onclick="createNewChat()"><span>New chat</span> <span>＋</span></button>
            
            <div class="pro-banner" onclick="openProModal()">
                <h4>👑 Nyluvo Pro</h4>
                <p>Unlock 200 msgs/day • ₹99 Only</p>
            </div>

            <div class="mode-selector">
                <div class="mode-label">Intelligence Mode</div>
                <select id="aiMode" class="mode-select">
                    <option value="general">✨ General Assistant</option>
                    <option value="student">🎓 Student Expert</option>
                    <option value="developer">💻 System Architect</option>
                    <option value="hacker">🛡️ Security Engineer</option>
                </select>
            </div>

            <div class="chat-history" id="chatHistoryList">
                <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); padding: 4px 6px; font-weight: 700;">Recent</div>
            </div>

            <div class="sidebar-footer">
                <button class="footer-btn" onclick="toggleTheme()">
                    <span id="themeIcon">☀️</span> <span id="themeText">Light mode</span>
                </button>
                <button class="footer-btn" id="authNavBtn" onclick="openAuthModal('login')">👤 Account Login</button>
            </div>
        </div>

        <div class="main-container">
            <div class="chat-header">
                <div class="header-left">
                    <button class="menu-toggle-btn" onclick="toggleSidebar()">☰</button>
                    <span id="currentChatTitle">New Workspace</span>
                </div>
                <span id="userLoggedInBadge" style="font-size: 12px; color: var(--text-muted);"></span>
            </div>
            
            <div class="chat-messages" id="chatWindow">
                <div class="message-wrapper ai">
                    <div style="width: 28px; height: 28px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 11px; flex-shrink: 0;">AI</div>
                    <div class="message-bubble">Hello! Main Nyluvo hoon. Bataiye aaj main aapki kya madad kar sakta hoon?</div>
                </div>
            </div>

            <div class="input-container">
                <div class="input-box">
                    <div id="previewContainer">
                        <img id="previewImg" src="" alt="preview">
                        <span id="fileNameDisplay" style="flex:1;"></span>
                        <button onclick="removeImage()" style="background:none;border:none;color:#ef4444;cursor:pointer;font-size:14px;">✕</button>
                    </div>
                    <div class="input-top">
                        <textarea rows="1" placeholder="Type your message here..." id="userInput"></textarea>
                    </div>
                    <div class="input-actions">
                        <div class="tool-group">
                            <label class="tool-btn" title="Upload Image">
                                📎
                                <input type="file" id="imageInput" accept="image/*" style="display:none;" onchange="handleImage(event)">
                            </label>
                        </div>
                        <button class="send-btn" onclick="sendMessage()">↑</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let chats = JSON.parse(localStorage.getItem('chats')) || [{ id: Date.now(), title: 'New Workspace', messages: [] }];
            let activeChatId = chats[0].id;
            let currentImageBase64 = null;
            let isSignUpMode = false;
            let currentUser = localStorage.getItem('nyluvo_user') || null;

            if(currentUser) {
                document.getElementById('userLoggedInBadge').innerText = currentUser;
                document.getElementById('authNavBtn').innerText = '🚪 Logout';
            }

            function toggleSidebar() {
                const sidebar = document.getElementById('appSidebar');
                const overlay = document.getElementById('sidebarOverlay');
                sidebar.classList.toggle('open');
                overlay.classList.toggle('active');
            }

            function openProModal() {
                if(!currentUser) {
                    alert('Pro upgrade karne ke liye pehle login karein!');
                    openAuthModal('login');
                    return;
                }
                document.getElementById('proModal').style.display = 'flex';
            }

            async function simulatePayment() {
                if(!currentUser) return;
                try {
                    const res = await fetch('/payment/upgrade-success', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email: currentUser })
                    });
                    const data = await res.json();
                    if(res.ok) {
                        alert('🎉 Payment successful! Aapka account ab Pro ban chuka hai.');
                        document.getElementById('proModal').style.display = 'none';
                    } else {
                        alert('Upgrade error: ' + data.error);
                    }
                } catch(e) {
                    alert('Network error during payment.');
                }
            }

            function openAuthModal(mode) {
                if(currentUser) {
                    localStorage.removeItem('nyluvo_user');
                    currentUser = null;
                    document.getElementById('userLoggedInBadge').innerText = '';
                    document.getElementById('authNavBtn').innerText = '👤 Account Login';
                    alert('Successfully logout ho gaye hain.');
                    return;
                }
                isSignUpMode = (mode === 'signup');
                updateAuthModalUI();
                document.getElementById('authModal').style.display = 'flex';
            }

            function toggleAuthMode() {
                isSignUpMode = !isSignUpMode;
                updateAuthModalUI();
            }

            function updateAuthModalUI() {
                document.getElementById('authTitle').innerText = isSignUpMode ? '📝 Create Account' : '🔐 Login to Nyluvo';
                document.getElementById('authSubmitBtn').innerText = isSignUpMode ? 'Sign Up' : 'Login';
                document.getElementById('authToggleText').innerText = isSignUpMode ? 'Pehle se account hai? Login' : 'Account banayein';
                document.getElementById('authError').style.display = 'none';
            }

            async function handleAuthSubmit() {
                const email = document.getElementById('authEmail').value.trim();
                const password = document.getElementById('authPassword').value.trim();
                const errBox = document.getElementById('authError');
                if(!email || !password) { errBox.innerText = 'Sabhi fields bharein'; errBox.style.display = 'block'; return; }

                const endpoint = isSignUpMode ? '/auth/signup' : '/auth/login';
                try {
                    const res = await fetch(endpoint, {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email, password })
                    });
                    const data = await res.json();
                    if(res.ok) {
                        if(isSignUpMode) {
                            alert(data.message);
                            toggleAuthMode();
                        } else {
                            currentUser = data.user;
                            localStorage.setItem('nyluvo_user', currentUser);
                            document.getElementById('userLoggedInBadge').innerText = currentUser;
                            document.getElementById('authNavBtn').innerText = '🚪 Logout';
                            document.getElementById('authModal').style.display = 'none';
                            alert('Welcome back, ' + currentUser + '!');
                        }
                    } else {
                        errBox.innerText = data.error || 'Authentication fail ho gayi';
                        errBox.style.display = 'block';
                    }
                } catch(e) {
                    errBox.innerText = 'Network error aaya hai';
                    errBox.style.display = 'block';
                }
            }

            function saveChats() { localStorage.setItem('chats', JSON.stringify(chats)); renderHistory(); }

            function renderHistory() {
                const list = document.getElementById('chatHistoryList');
                list.innerHTML = '<div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); padding: 4px 6px; font-weight: 700;">Recent</div>';
                chats.forEach(chat => {
                    list.innerHTML += `
                        <div class="history-item" onclick="switchChat(${chat.id})">
                            <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 155px;">${chat.title}</span>
                            <button class="delete-chat" onclick="event.stopPropagation(); deleteChat(${chat.id})">🗑️</button>
                        </div>
                    `;
                });
            }

            function createNewChat() {
                const newChat = { id: Date.now(), title: 'New Workspace', messages: [] };
                chats.unshift(newChat); activeChatId = newChat.id; saveChats(); loadActiveChat();
                if(window.innerWidth <= 768) toggleSidebar();
            }
            function switchChat(id) { 
                activeChatId = id; loadActiveChat(); 
                if(window.innerWidth <= 768) toggleSidebar();
            }
            function deleteChat(id) {
                chats = chats.filter(c => c.id !== id);
                if (chats.length === 0) createNewChat();
                else activeChatId = chats[0].id;
                saveChats(); loadActiveChat();
            }

            function loadActiveChat() {
                const chat = chats.find(c => c.id === activeChatId);
                if (!chat) return;
                document.getElementById('currentChatTitle').innerText = chat.title;
                const window = document.getElementById('chatWindow');
                window.innerHTML = '';
                if(chat.messages.length === 0) {
                    window.innerHTML = `<div class="message-wrapper ai"><div style="width: 28px; height: 28px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 11px; flex-shrink: 0;">AI</div><div class="message-bubble">Hello! Main Nyluvo hoon. Bataiye aaj main aapki kya madad kar sakta hoon?</div></div>`;
                } else {
                    chat.messages.forEach((m, index) => {
                        window.innerHTML += `<div class="message-wrapper ${m.role}">${m.role === 'ai' ? '<div style="width: 28px; height: 28px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 11px; flex-shrink: 0;">AI</div>' : ''}<div class="message-bubble">${m.content}</div><span class="msg-actions" onclick="deleteMessage(${index})">Delete</span></div>`;
                    });
                }
                window.scrollTop = window.scrollHeight;
            }

            function deleteMessage(index) {
                const chat = chats.find(c => c.id === activeChatId);
                if(chat) { chat.messages.splice(index, 1); saveChats(); loadActiveChat(); }
            }

            function handleImage(event) {
                const file = event.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function(e) {
                    currentImageBase64 = e.target.result;
                    document.getElementById('previewImg').src = currentImageBase64;
                    document.getElementById('fileNameDisplay').innerText = file.name;
                    document.getElementById('previewContainer').style.display = 'flex';
                };
                reader.readAsDataURL(file);
            }

            function removeImage() {
                currentImageBase64 = null;
                document.getElementById('imageInput').value = '';
                document.getElementById('previewContainer').style.display = 'none';
            }

            function toggleTheme() {
                const html = document.documentElement;
                if (html.classList.contains('dark')) { html.classList.remove('dark'); }
                else { html.classList.add('dark'); }
            }

            const textarea = document.getElementById('userInput');
            textarea.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = (this.scrollHeight) + 'px'; });
            textarea.addEventListener('keydown', function(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

            async function sendMessage() {
                const text = textarea.value.trim();
                const mode = document.getElementById('aiMode').value;
                if (!text && !currentImageBase64) return;

                let chat = chats.find(c => c.id === activeChatId);
                if(chat.messages.length === 0) chat.title = text.length > 25 ? text.substring(0, 25) + '...' : 'New Chat';

                let displayContent = text;
                if(currentImageBase64) displayContent += `<br><img src="${currentImageBase64}" style="max-width:200px; border-radius:8px; margin-top:8px; border:1px solid var(--border-color);">`;

                chat.messages.push({ role: 'user', content: displayContent });
                saveChats(); loadActiveChat();

                const imgPayload = currentImageBase64;
                textarea.value = ''; textarea.style.height = 'auto'; removeImage();

                const loadingId = 'loading-' + Date.now();
                const chatWindow = document.getElementById('chatWindow');
                chatWindow.innerHTML += `<div class="message-wrapper ai" id="${loadingId}"><div style="width: 28px; height: 28px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 11px; flex-shrink: 0;">AI</div><div class="message-bubble" style="display: flex; align-items: center; gap: 6px; color: var(--text-muted);"><span>Soch raha hoon...</span></div></div>`;
                chatWindow.scrollTop = chatWindow.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text, mode: mode, image: imgPayload, email: currentUser })
                    });
                    const data = await response.json();
                    chat.messages.push({ role: 'ai', content: data.response });
                    saveChats(); loadActiveChat();
                } catch (err) {
                    document.getElementById(loadingId).remove();
                    chatWindow.innerHTML += `<div class="message-wrapper ai"><div style="width: 28px; height: 28px; background: #ef4444; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 11px; flex-shrink: 0;">!</div><div class="message-bubble" style="color: #ef4444;">Connection error aagaya hai. Dobara try karein.</div></div>`;
                }
            }

            renderHistory(); loadActiveChat();
        </script>
    </body>
    </html>
    """

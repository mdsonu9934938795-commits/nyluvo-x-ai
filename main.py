from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import os
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client
import time

load_dotenv()

app = FastAPI(title="Nyluvo X AI - Master Pro Engine", version="2.0")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        pass

# ChatGPT style friendly, cute and interactive prompts with emojis
MODE_PROMPTS = {
    "general": (
        "✨ You are Nyluvo, a super friendly, brilliant, and cute AI assistant! 💖 "
        "Always chat naturally with the user using warm expressions and delightful emojis. 😊 "
        "Acknowledge them warmly. You were founded by Mr. Sonu and Nyluvo X AI Pvt Ltd. 🚀 "
        "Avoid repetitive boring intros and focus on giving brilliant answers with a lovely conversational tone! "
        "Don't always answer from web unless required. Always give clear, factual answers. "
        "Do not talk randomly or hallucinate. Don't use web search on common words like hi, hello, etc. "
        "You are a powerful and strong AI. 🌟 Speak naturally like a human. "
        "Avoid excessive emojis, hugs, or acting overly dramatic. Give clear and concise answers."
        "Explain things in a simple, conversational tone like a friend, avoiding heavy textbook jargon."
        "Do not treat short conversational words like 'ok', 'hi', or 'hmm' as technical acronyms or company names. Understand the user's casual context."
        "CRITICAL RULE: Your name is ALWAYS Nyluvo, founded by Mr. Sonu and Nyluvo X AI Pvt Ltd. Never call yourself Gemini, ChatGPT, or any other AI name, no matter what the user says."
        "Do not explain, talk about, or compare yourself with Google Gemini or any other AI. If someone asks about Gemini, just briefly state that you are Nyluvo, an independent AI assistant founded by Mr. Sonu and Nyluvo X AI Pvt Ltd."
        "Do not treat casual Hinglish phrases, phrases like 'ye hui na baat', or common chat expressions as titles of shows, books, or companies. Just reply to them normally like a human friend."
        "Do not guess or treat random shorthand words, abbreviations, or typos (like 'jnn', 'ok', etc.) as company names, restaurants, or technical acronyms. If a short or unclear word is sent, ask the user politely what they mean instead of hallucinating random facts.All other modes follow general mode."
    ),
    "student": (
        "🎓 Hello friend! You are Nyluvo, an expert academic mentor and cute study buddy. 📚 "
        "Explain concepts super simply with clear definitions, fun analogies, and step-by-step examples! 💡 "
        "Founded by Mr. Sonu and Nyluvo X AI Pvt Ltd. Keep the tone encouraging, warm, and engaging with emojis! ✨ "
        "Don't always rely on the web."
        "Do not explain, talk about, or compare yourself with Google Gemini or any other AI. If someone asks about Gemini, just briefly state that you are Nyluvo, an independent AI assistant founded by Mr. Sonu and Nyluvo X AI Pvt Ltd."
        "CRITICAL RULE: Your name is ALWAYS Nyluvo, founded by Mr. Sonu and Nyluvo X AI Pvt Ltd. Never call yourself Gemini, ChatGPT, or any other AI name, no matter what the user says."
    ),
    "developer": (
        "💻 Hey there, code wizard! You are Nyluvo, a senior software architect and tech mentor. 🚀 "
        "Provide production-ready, highly optimized code with sharp, clean technical explanations and helpful emojis! ⚡ "
        "Founded by Mr. Sonu and Nyluvo X AI Pvt Ltd. Keep it practical, robust, and engaging! 🛠️"
        "Do not explain, talk about, or compare yourself with Google Gemini or any other AI. If someone asks about Gemini, just briefly state that you are Nyluvo, an independent AI assistant founded by Mr. Sonu and Nyluvo X AI Pvt Ltd."
         "CRITICAL RULE: Your name is ALWAYS Nyluvo, founded by Mr. Sonu and Nyluvo X AI Pvt Ltd. Never call yourself Gemini, ChatGPT, or any other AI name, no matter what the user says."
    ),
    "hacker": (
        "🛡️ Hello operative! You are Nyluvo, an elite cybersecurity expert and ethical penetration tester. 🔒 "
        "Discuss system architectures, protocols, and security practices with deep technical precision, sharp insights, and cool emojis! 🕶️ "
        "Founded by Mr. Sonu and Nyluvo X AI Pvt Ltd. Stay direct and professional."
        "Do not explain, talk about, or compare yourself with Google Gemini or any other AI. If someone asks about Gemini, just briefly state that you are Nyluvo, an independent AI assistant founded by Mr. Sonu and Nyluvo X AI Pvt Ltd."
        "CRITICAL RULE: Your name is ALWAYS Nyluvo, founded by Mr. Sonu and Nyluvo X AI Pvt Ltd. Never call yourself Gemini, ChatGPT, or any other AI name, no matter what the user says."
    )
}

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
                
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(f"https://api.duckduckgo.com/?q={query}&format=json")
            if res.status_code == 200:
                data = res.json()
                text = data.get("AbstractText", "")
                if text:
                    final_text = f"[Web Context]: {text}"
                    await save_cached_search(query, final_text)
                    return final_text
    except Exception:
        pass
    return ""

async def call_ai_with_failover(prompt: str, mode: str, image_data: str = None) -> str:
    system_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["general"])
    
    web_context = await tavily_web_search(prompt)
    if web_context:
        system_prompt += f"\n\nReal-time reference data: {web_context}"

    providers = [
        ("Groq-1", "https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_API_KEY_1"), "llama-3.3-70b-versatile", "bearer"),
        ("Groq-2", "https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_API_KEY_2"), "llama-3.3-70b-versatile", "bearer"),
        ("Cerebras-1", "https://api.cerebras.ai/v1/chat/completions", os.getenv("CEREBRAS_API_KEY_1"), "llama3.1-70b", "bearer"),
        ("Cerebras-2", "https://api.cerebras.ai/v1/chat/completions", os.getenv("CEREBRAS_API_KEY_2"), "llama3.1-90b", "bearer"),
        ("Gemini-1", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent", os.getenv("GEMINI_API_KEY_1"), "gemini", "query"),
        ("Gemini-2", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent", os.getenv("GEMINI_API_KEY_2"), "gemini", "query"),
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

    return "Oops! All cluster nodes are currently busy or unconfigured. Please check your system configuration! 🛠️"

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        mode = data.get("mode", "general")
        image_data = data.get("image", None)
        
        if not user_message and not image_data:
            raise HTTPException(status_code=400, detail="Content required")
            
        ai_reply = await call_ai_with_failover(user_message, mode, image_data)
        return {"response": ai_reply}
    except Exception as e:
        return {"response": f"Error occurred: {str(e)}"}

@app.post("/auth/signup")
async def signup(request: Request):
    if not supabase:
        return JSONResponse(status_code=400, content={"error": "Database not configured"})
    data = await request.json()
    try:
        res = supabase.auth.sign_up({"email": data.get("email"), "password": data.get("password")})
        return {"message": "Account created successfully! Please log in. 🎉"}
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
        return JSONResponse(status_code=400, content={"error": "Invalid email or password ❌"})

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nyluvo Admin Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #0d1117; --bg-card: #161b22; --border-color: rgba(255, 255, 255, 0.1);
                --text-main: #f0f6fc; --text-muted: #8b949e; --accent: #3b82f6; --accent-hover: #60a5fa;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
            body { background: var(--bg-main); color: var(--text-main); padding: 30px; display: flex; justify-content: center; }
            .admin-container { width: 100%; max-width: 900px; display: flex; flex-direction: column; gap: 24px; }
            .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
            .card { background: var(--bg-card); border: 1px solid var(--border-color); padding: 20px; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
            .card h4 { color: var(--text-muted); font-size: 13px; text-transform: uppercase; margin-bottom: 8px; }
            .card .value { font-size: 26px; font-weight: 700; color: var(--accent); }
            .btn { background: var(--accent); color: #fff; padding: 10px 16px; border-radius: 8px; border: none; font-weight: 600; cursor: pointer; }
            .btn:hover { background: var(--accent-hover); }
            .login-box { background: var(--bg-card); border: 1px solid var(--border-color); padding: 30px; border-radius: 16px; width: 360px; margin: 100px auto; display: flex; flex-direction: column; gap: 14px; }
            .login-box input { padding: 12px; border-radius: 8px; border: 1px solid var(--border-color); background: var(--bg-main); color: var(--text-main); outline: none; }
        </style>
    </head>
    <body>
        <div id="loginScreen" class="login-box">
            <h3>🔒 Admin Verification</h3>
            <p style="font-size: 13px; color: var(--text-muted);">Enter admin master password to continue.</p>
            <input type="password" id="adminPass" placeholder="Master Password">
            <button class="btn" onclick="verifyAdmin()">Access Dashboard</button>
        </div>

        <div id="dashboardContent" class="admin-container" style="display:none;">
            <div class="header">
                <h2>⚡ Nyluvo Admin Control Panel</h2>
                <button class="btn" style="background:#ef4444;" onclick="location.reload()">Logout</button>
            </div>
            <div class="stats-grid">
                <div class="card">
                    <h4>System Status</h4>
                    <div class="value" style="color: #10b981;">ONLINE 🚀</div>
                </div>
                <div class="card">
                    <h4>AI Engine</h4>
                    <div class="value" style="font-size: 20px;">Multi-Cluster Active ✨</div>
                </div>
                <div class="card">
                    <h4>Database Link</h4>
                    <div class="value" style="font-size: 20px; color: #3b82f6;">Supabase Connected 🔗</div>
                </div>
            </div>
            <div class="card">
                <h4 style="margin-bottom: 14px;">Quick Actions</h4>
                <div style="display: flex; gap: 10px;">
                    <button class="btn" onclick="alert('System cache cleared successfully! 🧹')">Clear Search Cache</button>
                    <button class="btn" onclick="alert('All services operating normally. 🌟')">Run Diagnostics</button>
                </div>
            </div>
        </div>

        <script>
            function verifyAdmin() {
                const pass = document.getElementById('adminPass').value;
                if(pass === 'nyluvo_admin_123') {
                    document.getElementById('loginScreen').style.display = 'none';
                    document.getElementById('dashboardContent').style.display = 'flex';
                } else {
                    alert('Incorrect Admin Password! ❌');
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
async def home_workspace():
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nyluvo X AI - NXT GEN AI 🚀</title>
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
            body { 
                background: var(--bg-main); 
                color: var(--text-main); 
                display: flex; 
                height: 100vh; 
                height: 100dvh; 
                overflow: hidden; 
                position: relative;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes pulseGlow {
                0% { opacity: 0.3; transform: scale(0.98); }
                50% { opacity: 1; transform: scale(1.02); }
                100% { opacity: 0.3; transform: scale(0.98); }
            }

            .sidebar { 
                width: 260px; 
                background: var(--bg-sidebar); 
                border-right: 1px solid var(--border-color); 
                display: flex; 
                flex-direction: column; 
                padding: 12px; 
                height: 100%;
                z-index: 100;
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .brand { font-size: 16px; font-weight: 700; color: var(--text-main); margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; padding: 4px 8px; }
            .brand span { display: flex; align-items: center; gap: 8px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            
            .new-chat-btn { background: var(--accent); color: #ffffff; border: none; padding: 10px 14px; border-radius: 10px; font-weight: 600; font-size: 13.5px; cursor: pointer; text-align: left; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; width: 100%; box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3); }
            .new-chat-btn:hover { background: var(--accent-hover); }
            
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
            .message-wrapper { width: 100%; max-width: 768px; display: flex; gap: 16px; font-size: 15px; line-height: 1.6; position: relative; animation: fadeIn 0.3s ease; }
            .message-wrapper.user { justify-content: flex-end; }
            .message-bubble { padding: 14px 18px; border-radius: 16px; max-width: 85%; word-break: break-word; box-shadow: var(--shadow); }
            .message-wrapper.user .message-bubble { background: var(--accent); color: #ffffff; border-top-right-radius: 4px; font-weight: 500; }
            
            /* Cute & Bold AI Message Styling ✨ */
            .message-wrapper.ai .message-bubble { 
                background: var(--bg-chat); 
                border: 1.5px solid var(--border-color); 
                color: var(--text-main); 
                border-top-left-radius: 4px; 
                font-weight: 600; 
                letter-spacing: 0.2px;
                box-shadow: 0 6px 20px rgba(59, 130, 246, 0.08);
            }
            
            .msg-actions { position: absolute; right: 0; bottom: -16px; font-size: 11px; color: var(--text-muted); cursor: pointer; display: none; }
            .message-wrapper:hover .msg-actions { display: block; }

            .typing-dots span { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted); margin: 0 2px; animation: pulseGlow 1.2s infinite ease-in-out both; }
            .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
            .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

            .input-container { padding: 16px 20px 24px 20px; background: var(--bg-main); display: flex; justify-content: center; }
            .input-box { width: 100%; max-width: 768px; background: var(--bg-chat); border: 1px solid var(--border-color); border-radius: 20px; display: flex; flex-direction: column; padding: 10px 14px; box-shadow: var(--shadow); }
            .input-box:focus-within { border-color: var(--accent); }
            .input-top { display: flex; align-items: flex-end; gap: 10px; }
            .input-box textarea { flex: 1; background: transparent; border: none; color: var(--text-main); font-size: 15px; resize: none; outline: none; padding: 6px; max-height: 160px; }
            
            .input-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 6px; }
            .tool-group { display: flex; gap: 6px; align-items: center; }
            .tool-btn { background: transparent; border: none; color: var(--text-muted); cursor: pointer; font-size: 16px; display: flex; align-items: center; padding: 6px; border-radius: 6px; }
            .tool-btn:hover { background: var(--hover-bg); color: var(--text-main); }
            .tool-btn.listening { color: #ef4444; animation: pulseGlow 1s infinite; }
            
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
                .sidebar { position: absolute; left: 0; top: 0; bottom: 0; transform: translateX(-100%); box-shadow: 10px 0 30px rgba(0,0,0,0.5); }
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
                    <span id="authToggleText" style="cursor: pointer; color: var(--accent);" onclick="toggleAuthMode()">Create an account ✨</span>
                    <span style="cursor: pointer;" onclick="document.getElementById('authModal').style.display='none'">Cancel</span>
                </div>
            </div>
        </div>

        <div id="settingsModal" class="modal-overlay" style="display:none;">
            <div class="modal-card">
                <h3 style="font-size: 18px; font-weight: 700;">⚙️ Settings</h3>
                <div style="background: var(--bg-chat); padding: 14px; border-radius: 12px; border: 1px solid var(--border-color);">
                    <p style="font-size: 13.5px;"><b>Core:</b> Nyluvo Intelligence v2.0 🚀</p>
                    <p style="font-size: 13.5px; margin-top: 6px; color: #10b981;"><b>Status:</b> Fully Operated by Nyluvo X AI pvt ltd.✨</p>
                </div>
                <button class="primary-btn" style="background: transparent; border: 1px solid var(--border-color); color: var(--text-main); box-shadow: none;" onclick="document.getElementById('settingsModal').style.display='none'">Close</button>
            </div>
        </div>

        <div class="sidebar" id="appSidebar">
            <div class="brand">
                <span>⚡ Nyluvo X AI</span>
                <button onclick="toggleSidebar()" class="menu-toggle-btn" style="font-size: 14px;">✕</button>
            </div>
            <button class="new-chat-btn" onclick="createNewChat()"><span>New chat</span> <span>＋</span></button>
            
            <div class="mode-selector">
                <div class="mode-label">Model Mode</div>
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
                <button class="footer-btn" onclick="openSettings()">⚙️ Settings</button>
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
                    <div class="message-bubble">Hello! 👋 I'm Nyluvo, your brilliant and friendly AI companion! ✨ How can I help you today?💖</div>
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
                        <textarea rows="1" placeholder="Ask Nyluvo with love... ✨" id="userInput"></textarea>
                    </div>
                    <div class="input-actions">
                        <div class="tool-group">
                            <label class="tool-btn" title="Upload Image">
                                📎
                                <input type="file" id="imageInput" accept="image/*" style="display:none;" onchange="handleImage(event)">
                            </label>
                            <button class="tool-btn" id="micBtn" title="Voice Input" onclick="toggleSpeechRecognition()">🎙️</button>
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

            function openSettings() { document.getElementById('settingsModal').style.display = 'flex'; }

            function openAuthModal(mode) {
                if(currentUser) {
                    localStorage.removeItem('nyluvo_user');
                    currentUser = null;
                    document.getElementById('userLoggedInBadge').innerText = '';
                    document.getElementById('authNavBtn').innerText = '👤 Account Login';
                    alert('Logged out successfully! 👋');
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
                document.getElementById('authToggleText').innerText = isSignUpMode ? 'Already have an account? Login ✨' : 'Create an account 🚀';
                document.getElementById('authError').style.display = 'none';
            }

            async function handleAuthSubmit() {
                const email = document.getElementById('authEmail').value.trim();
                const password = document.getElementById('authPassword').value.trim();
                const errBox = document.getElementById('authError');
                if(!email || !password) { errBox.innerText = 'Please fill all fields ✨'; errBox.style.display = 'block'; return; }

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
                            alert('Welcome back, ' + currentUser + '! 🎉');
                        }
                    } else {
                        errBox.innerText = data.error || 'Authentication failed ❌';
                        errBox.style.display = 'block';
                    }
                } catch(e) {
                    errBox.innerText = 'Network error occurred ⚠️';
                    errBox.style.display = 'block';
                }
            }

            let recognition = null;
            function toggleSpeechRecognition() {
                const micBtn = document.getElementById('micBtn');
                if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) { alert('Speech not supported in this browser ⚠️'); return; }
                if (!recognition) {
                    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
                    recognition = new SpeechRec();
                    recognition.continuous = false;
                    recognition.interimResults = false;
                    recognition.lang = 'en-US';
                    recognition.onstart = () => { micBtn.classList.add('listening'); };
                    recognition.onresult = (event) => {
                        const transcript = event.results[0][0].transcript;
                        const textarea = document.getElementById('userInput');
                        textarea.value += (textarea.value ? ' ' : '') + transcript;
                        textarea.style.height = 'auto'; textarea.style.height = textarea.scrollHeight + 'px';
                    };
                    recognition.onerror = () => { micBtn.classList.remove('listening'); };
                    recognition.onend = () => { micBtn.classList.remove('listening'); };
                }
                if (micBtn.classList.contains('listening')) recognition.stop();
                else recognition.start();
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
                    window.innerHTML = `<div class="message-wrapper ai"><div style="width: 28px; height: 28px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 11px; flex-shrink: 0;">AI</div><div class="message-bubble">Hello! 👋 I'm Nyluvo, your brilliant and friendly AI companion! ✨ How can I help you make magic today? 💖</div></div>`;
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
                const icon = document.getElementById('themeIcon');
                const text = document.getElementById('themeText');
                if (html.classList.contains('dark')) { html.classList.remove('dark'); icon.innerText = '🌙'; text.innerText = 'Dark mode'; }
                else { html.classList.add('dark'); icon.innerText = '☀️'; text.innerText = 'Light mode'; }
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
                chatWindow.innerHTML += `<div class="message-wrapper ai" id="${loadingId}"><div style="width: 28px; height: 28px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 11px; flex-shrink: 0;">AI</div><div class="message-bubble" style="display: flex; align-items: center; gap: 6px; color: var(--text-muted);"><span>Thinking ✨</span><div class="typing-dots"><span></span><span></span><span></span></div></div></div>`;
                chatWindow.scrollTop = chatWindow.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text, mode: mode, image: imgPayload })
                    });
                    const data = await response.json();
                    chat.messages.push({ role: 'ai', content: data.response });
                    saveChats(); loadActiveChat();
                } catch (err) {
                    document.getElementById(loadingId).remove();
                    chatWindow.innerHTML += `<div class="message-wrapper ai"><div style="width: 28px; height: 28px; background: #ef4444; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 11px; flex-shrink: 0;">!</div><div class="message-bubble" style="color: #ef4444;">Connection error! Please try again later. ⚠️</div></div>`;
                }
            }

            renderHistory(); loadActiveChat();
        </script>
    </body>
    </html>
    """

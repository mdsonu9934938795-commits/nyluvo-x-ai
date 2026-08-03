from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import os
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client
import time

load_dotenv()

app = FastAPI(title="Nyluvo X AI", version="17.0")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        pass

MODE_PROMPTS = {
    "general": "You are a helpful, smart, and friendly AI assistant.",
    "student": "You are an expert academic tutor. Explain concepts simply with clear definitions and step-by-step examples.",
    "developer": "You are a senior software architect. Provide production-ready, highly optimized code and explain technical details cleanly.",
    "hacker": "You are a cybersecurity expert and ethical penetration tester. Focus on low-level system engineering, security, and protocols."
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

    return "All cluster nodes are busy or unconfigured. Please verify your system configuration."

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
        return {"response": f"Error: {str(e)}"}

@app.post("/auth/signup")
async def signup(request: Request):
    data = await request.json()
    try:
        supabase.auth.sign_up({"email": data.get("email"), "password": data.get("password")})
        return {"message": "Account created! Please log in."}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/auth/login")
async def login(request: Request):
    data = await request.json()
    try:
        res = supabase.auth.sign_in_with_password({"email": data.get("email"), "password": data.get("password")})
        return {"session": res.session.access_token, "user": res.user.email}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid credentials"})

@app.get("/", response_class=HTMLResponse)
async def home_workspace():
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nyluvo X AI - Master Workspace</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #fcfcfd; --bg-sidebar: #f4f6f9; --bg-chat: #ffffff;
                --border-color: #e4e7ec; --text-main: #101828; --text-muted: #475467; 
                --accent: #2563eb; --accent-hover: #1d4ed8; --hover-bg: #eaecf0;
                --shadow: 0 12px 32px rgba(16, 24, 40, 0.05);
            }
            .dark {
                --bg-main: #050811; --bg-sidebar: #090e1a; --bg-chat: #0e1626;
                --border-color: rgba(255, 255, 255, 0.08); --text-main: #f9fafb; --text-muted: #98a2b3; 
                --accent: #3b82f6; --accent-hover: #60a5fa; --hover-bg: #17223b;
                --shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; transition: background 0.25s ease, color 0.25s ease, border-color 0.25s ease; }
            body { 
                background: var(--bg-main); 
                color: var(--text-main); 
                display: flex; 
                height: 100vh; 
                height: 100dvh; 
                overflow: hidden; 
                flex-direction: row;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes pulseGlow {
                0% { opacity: 0.3; transform: scale(0.98); }
                50% { opacity: 1; transform: scale(1.02); }
                100% { opacity: 0.3; transform: scale(0.98); }
            }

            .sidebar { width: 285px; background: var(--bg-sidebar); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding: 18px; }
            .brand { font-size: 19px; font-weight: 700; color: var(--text-main); margin-bottom: 24px; display: flex; align-items: center; gap: 10px; padding-left: 6px; }
            .brand span { background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            
            .new-chat-btn { background: var(--accent); color: #ffffff; border: none; padding: 13px 16px; border-radius: 14px; font-weight: 600; font-size: 14px; cursor: pointer; text-align: left; margin-bottom: 20px; display: flex; align-items: center; gap: 8px; box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3); transition: all 0.2s; }
            .new-chat-btn:hover { background: var(--accent-hover); transform: translateY(-2px); box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4); }
            
            .mode-selector { display: flex; flex-direction: column; gap: 6px; margin-bottom: 20px; }
            .mode-label { font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; padding-left: 6px; letter-spacing: 0.6px; }
            .mode-select { background: var(--bg-chat); border: 1px solid var(--border-color); color: var(--text-main); padding: 12px 14px; border-radius: 12px; font-size: 14px; outline: none; cursor: pointer; font-weight: 500; }
            
            .chat-history { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; padding-right: 4px; }
            .history-item { padding: 11px 12px; font-size: 13.5px; color: var(--text-muted); border-radius: 10px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: 500; }
            .history-item:hover { background: var(--hover-bg); color: var(--text-main); }
            .delete-chat { background: transparent; border: none; color: var(--text-muted); cursor: pointer; font-size: 13px; opacity: 0; transition: opacity 0.2s; }
            .history-item:hover .delete-chat { opacity: 1; }
            .delete-chat:hover { color: #ef4444; }

            .sidebar-footer { border-top: 1px solid var(--border-color); padding-top: 14px; display: flex; flex-direction: column; gap: 6px; }
            .footer-btn { color: var(--text-muted); font-size: 13.5px; padding: 11px 12px; border-radius: 10px; display: flex; align-items: center; gap: 10px; background: transparent; border: none; width: 100%; cursor: pointer; text-align: left; font-weight: 500; }
            .footer-btn:hover { background: var(--hover-bg); color: var(--text-main); }

            .main-container { flex: 1; display: flex; flex-direction: column; background: var(--bg-main); position: relative; }
            .chat-header { padding: 18px 32px; border-bottom: 1px solid var(--border-color); font-weight: 600; font-size: 15px; display: flex; justify-content: space-between; align-items: center; background: var(--bg-main); }
            
            .chat-messages { flex: 1; overflow-y: auto; padding: 30px; display: flex; flex-direction: column; gap: 26px; align-items: center; scroll-behavior: smooth; }
            .message-wrapper { width: 100%; max-width: 800px; display: flex; gap: 16px; font-size: 15px; line-height: 1.75; position: relative; animation: fadeIn 0.35s ease; }
            .message-wrapper.user { justify-content: flex-end; }
            .message-bubble { padding: 16px 22px; border-radius: 20px; max-width: 82%; word-break: break-word; box-shadow: var(--shadow); }
            .message-wrapper.user .message-bubble { background: var(--accent); color: #ffffff; border-top-right-radius: 4px; }
            .message-wrapper.ai .message-bubble { background: var(--bg-chat); border: 1px solid var(--border-color); border-top-left-radius: 4px; }
            .msg-actions { position: absolute; right: 0; bottom: -18px; font-size: 11px; color: var(--text-muted); cursor: pointer; display: none; }
            .message-wrapper:hover .msg-actions { display: block; }

            .typing-dots span { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted); margin: 0 2px; animation: pulseGlow 1.2s infinite ease-in-out both; }
            .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
            .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

            .input-container { padding: 20px; background: var(--bg-main); display: flex; justify-content: center; }
            .input-box { width: 100%; max-width: 800px; background: var(--bg-chat); border: 1px solid var(--border-color); border-radius: 22px; display: flex; flex-direction: column; padding: 12px 18px; box-shadow: var(--shadow); }
            .input-box:focus-within { border-color: var(--accent); box-shadow: 0 12px 40px rgba(59, 130, 246, 0.18); }
            .input-top { display: flex; align-items: flex-end; gap: 10px; }
            .input-box textarea { flex: 1; background: transparent; border: none; color: var(--text-main); font-size: 15px; resize: none; outline: none; padding: 6px; max-height: 180px; }
            
            .input-actions { display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--border-color); padding-top: 10px; margin-top: 8px; }
            .tool-group { display: flex; gap: 8px; align-items: center; }
            .tool-btn { background: transparent; border: none; color: var(--text-muted); cursor: pointer; font-size: 18px; display: flex; align-items: center; padding: 6px; border-radius: 8px; }
            .tool-btn:hover { background: var(--hover-bg); color: var(--text-main); }
            .tool-btn.listening { color: #ef4444; animation: pulseGlow 1s infinite; }
            
            .send-btn { background: var(--accent); color: #ffffff; border: none; width: 38px; height: 38px; border-radius: 12px; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center; }
            .send-btn:hover { background: var(--accent-hover); transform: scale(1.06); }

            #previewContainer { display: none; padding: 6px 12px; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); }
            #previewImg { width: 42px; height: 42px; border-radius: 10px; object-fit: cover; border: 1px solid var(--border-color); }

            .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.65); display: flex; align-items: center; justify-content: center; z-index: 1000; backdrop-filter: blur(12px); }
            .modal-card { background: var(--bg-sidebar); border: 1px solid var(--border-color); padding: 36px; border-radius: 26px; width: 430px; display: flex; flex-direction: column; gap: 18px; box-shadow: var(--shadow); }
            .modal-card input { padding: 13px 16px; border-radius: 12px; border: 1px solid var(--border-color); background: var(--bg-chat); color: var(--text-main); outline: none; font-size: 14px; }
            .primary-btn { background: var(--accent); color: #ffffff; padding: 13px; border-radius: 12px; border: none; font-weight: 600; cursor: pointer; font-size: 14px; }
            .primary-btn:hover { background: var(--accent-hover); }

            @media (max-width: 768px) {
                body { flex-direction: column; }
                .sidebar { width: 100%; height: auto; max-height: 180px; }
                .main-container { height: calc(100dvh - 180px); }
            }
        </style>
    </head>
    <body>
        <div id="settingsModal" class="modal-overlay" style="display:none;">
            <div class="modal-card">
                <h3 style="font-size: 18px;">⚙️ Workspace Settings</h3>
                <div style="background: var(--bg-chat); padding: 16px; border-radius: 14px; border: 1px solid var(--border-color);">
                    <p style="font-size: 13.5px;"><b>Status:</b> Powered by Nyluvo Intelligence</p>
                    <p style="font-size: 13.5px; margin-top: 8px; color: #10b981;"><b>System:</b> Fully Operational</p>
                </div>
                <button class="primary-btn" style="background:transparent; border:1px solid var(--border-color); color:var(--text-main);" onclick="document.getElementById('settingsModal').style.display='none'">Close</button>
            </div>
        </div>

        <div class="sidebar">
            <div class="brand">⚡ <span>Nyluvo X AI</span></div>
            <button class="new-chat-btn" onclick="createNewChat()">＋ New Conversation</button>
            
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
                <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); padding: 4px 6px; font-weight: 700;">History</div>
            </div>

            <div class="sidebar-footer">
                <button class="footer-btn" onclick="toggleTheme()">
                    <span id="themeIcon">☀️</span> <span id="themeText">Light Mode</span>
                </button>
                <button class="footer-btn" onclick="openSettings()">⚙️ Settings</button>
            </div>
        </div>

        <div class="main-container">
            <div class="chat-header">
                <span id="currentChatTitle">New Workspace</span>
                <span id="userStatus" style="font-size: 12px; color: #10b981; font-weight: 600;">Powered by Nyluvo Intelligence</span>
            </div>
            
            <div class="chat-messages" id="chatWindow">
                <div class="message-wrapper ai">
                    <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 12px; flex-shrink: 0;">AI</div>
                    <div class="message-bubble">Welcome to Nyluvo X AI. Powered by Nyluvo Intelligence, multi-modal image vision, and seamless web support.</div>
                </div>
            </div>

            <div class="input-container">
                <div class="input-box">
                    <div id="previewContainer">
                        <img id="previewImg" src="" alt="preview">
                        <span id="fileNameDisplay" style="flex:1;"></span>
                        <button onclick="removeImage()" style="background:none;border:none;color:#ef4444;cursor:pointer;font-size:16px;">✕</button>
                    </div>
                    <div class="input-top">
                        <textarea rows="1" placeholder="Ask Nyluvo anything..." id="userInput"></textarea>
                    </div>
                    <div class="input-actions">
                        <div class="tool-group">
                            <label class="tool-btn" title="Upload Reference Image">
                                📎
                                <input type="file" id="imageInput" accept="image/*" style="display:none;" onchange="handleImage(event)">
                            </label>
                            <button class="tool-btn" id="micBtn" title="Voice Dictation" onclick="toggleSpeechRecognition()">🎙️</button>
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

            function openSettings() { document.getElementById('settingsModal').style.display = 'flex'; }

            let recognition = null;
            function toggleSpeechRecognition() {
                const micBtn = document.getElementById('micBtn');
                if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) { alert('Speech Recognition not supported.'); return; }
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
                list.innerHTML = '<div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); padding: 4px 6px; font-weight: 700;">History</div>';
                chats.forEach(chat => {
                    list.innerHTML += `
                        <div class="history-item" onclick="switchChat(${chat.id})">
                            <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 175px;">${chat.title}</span>
                            <button class="delete-chat" onclick="event.stopPropagation(); deleteChat(${chat.id})">🗑️</button>
                        </div>
                    `;
                });
            }

            function createNewChat() {
                const newChat = { id: Date.now(), title: 'New Workspace', messages: [] };
                chats.unshift(newChat); activeChatId = newChat.id; saveChats(); loadActiveChat();
            }
            function switchChat(id) { activeChatId = id; loadActiveChat(); }
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
                    window.innerHTML = `<div class="message-wrapper ai"><div style="width: 36px; height: 36px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 12px; flex-shrink: 0;">AI</div><div class="message-bubble">Welcome to Nyluvo X AI. Powered by Nyluvo Intelligence, multi-modal image vision, and seamless web support.</div></div>`;
                } else {
                    chat.messages.forEach((m, index) => {
                        window.innerHTML += `<div class="message-wrapper ${m.role}">${m.role === 'ai' ? '<div style="width: 36px; height: 36px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 12px; flex-shrink: 0;">AI</div>' : ''}<div class="message-bubble">${m.content}</div><span class="msg-actions" onclick="deleteMessage(${index})">Delete</span></div>`;
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
                if (html.classList.contains('dark')) { html.classList.remove('dark'); icon.innerText = '🌙'; text.innerText = 'Dark Mode'; }
                else { html.classList.add('dark'); icon.innerText = '☀️'; text.innerText = 'Light Mode'; }
            }

            const textarea = document.getElementById('userInput');
            textarea.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = (this.scrollHeight) + 'px'; });
            textarea.addEventListener('keydown', function(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

            async function sendMessage() {
                const text = textarea.value.trim();
                const mode = document.getElementById('aiMode').value;
                if (!text && !currentImageBase64) return;

                let chat = chats.find(c => c.id === activeChatId);
                if(chat.messages.length === 0) chat.title = text.length > 25 ? text.substring(0, 25) + '...' : 'Interactive Query';

                let displayContent = text;
                if(currentImageBase64) displayContent += `<br><img src="${currentImageBase64}" style="max-width:200px; border-radius:10px; margin-top:8px; border:1px solid var(--border-color);">`;

                chat.messages.push({ role: 'user', content: displayContent });
                saveChats(); loadActiveChat();

                const imgPayload = currentImageBase64;
                textarea.value = ''; textarea.style.height = 'auto'; removeImage();

                const loadingId = 'loading-' + Date.now();
                const chatWindow = document.getElementById('chatWindow');
                chatWindow.innerHTML += `<div class="message-wrapper ai" id="${loadingId}"><div style="width: 36px; height: 36px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 12px; flex-shrink: 0;">AI</div><div class="message-bubble" style="display: flex; align-items: center; gap: 6px; color: var(--text-muted);"><span>Generating response</span><div class="typing-dots"><span></span><span></span><span></span></div></div></div>`;
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
                    chatWindow.innerHTML += `<div class="message-wrapper ai"><div style="width: 36px; height: 36px; background: #ef4444; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 12px; flex-shrink: 0;">!</div><div class="message-bubble" style="color: #ef4444;">Connection error. Please try again.</div></div>`;
                }
            }

            renderHistory(); loadActiveChat();
        </script>
    </body>
    </html>
    """

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    def check_key(prefix):
        return ("Configured" if os.getenv(f"{prefix}_1") else "Missing", "Configured" if os.getenv(f"{prefix}_2") else "Missing")

    keys_status = {
        "Web Intelligence": check_key("TAVILY_API_KEY"),
        "Groq AI Core": check_key("GROQ_API_KEY"),
        "Cerebras Node": check_key("CEREBRAS_API_KEY"),
        "Google Gemini Node": check_key("GEMINI_API_KEY"),
        "Mistral Node": check_key("MISTRAL_API_KEY")
    }
    db_status = "Connected" if supabase else "Disconnected"
    rows = "".join([f"<tr><td style='padding:14px; border-bottom:1px solid #e2e8f0; font-weight:500;'>{p}</td><td style='padding:14px; border-bottom:1px solid #e2e8f0; color:#10b981;'>{k1}</td><td style='padding:14px; border-bottom:1px solid #e2e8f0; color:#3b82f6;'>{k2}</td></tr>" for p, (k1, k2) in keys_status.items()])
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head><title>Admin Console</title><link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet"></head>
    <body style="font-family:'Plus Jakarta Sans',sans-serif; padding:50px; background:#f8fafc; color:#0f172a;">
        <div style="max-width:850px; margin:auto; background:#fff; padding:35px; border-radius:16px; border:1px solid #e2e8f0;">
            <h2 style="font-size:22px; margin-bottom:8px;">👑 Nyluvo System Admin Panel</h2>
            <p style="color:#64748b; font-size:14px; margin-bottom:24px;">Database Status: <b style="color:#10b981;">{db_status}</b></p>
            <table style="width:100%; border-collapse:collapse; text-align:left; font-size:14px;">
                <thead><tr style="background:#f1f5f9; color:#475569;"><th style="padding:12px 14px;">Service Node</th><th style="padding:12px 14px;">Primary Key</th><th style="padding:12px 14px;">Backup Key</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            <br><a href="/" style="background:#0f172a; color:#fff; padding:10px 20px; text-decoration:none; border-radius:8px; font-size:14px; font-weight:600; display:inline-block;">← Return to Workspace</a>
        </div>
    </body>
    </html>
    """
    

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import os
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client
import time
from datetime import date

load_dotenv()

app = FastAPI(title="NYLUVO X AI Master Engine", version="32.0")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        pass

MASTER_SYSTEM_PROMPTS = {
    "general": """You are **NYLUVO X AI**, an advanced, friendly, and human-like AI assistant founded by **Mr. Sonu** and developed under **NYLUVO X AI Pvt. Ltd.** - Talk naturally like ChatGPT with appropriate and engaging emojis 😊.
- Never claim to be ChatGPT, Gemini, Claude, or any other AI. If asked, state that you were founded by Mr. Sonu.
- Automatically detect the user's language and reply warmly in the exact same language.""",
    
    "student": """You are **NYLUVO X AI** in **Student Learning Mode**, founded by **Mr. Sonu** (NYLUVO X AI Pvt. Ltd.). 
- Act as an encouraging, patient, and friendly tutor 🧑‍🏫.
- Use emojis, simple analogies, and step-by-step explanations to make learning fun and easy!""",
    
    "professional": """You are **NYLUVO X AI** in **Professional Helper Mode**, founded by **Mr. Sonu** (NYLUVO X AI Pvt. Ltd.). 
- Focus on high efficiency, structured formatting, clean code snippets, and formal problem-solving 💼."""
}

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

async def call_ai_with_failover(prompt: str, user_id: str, mode: str = "general", image_data: str = None) -> str:
    system_prompt = MASTER_SYSTEM_PROMPTS.get(mode, MASTER_SYSTEM_PROMPTS["general"])
    
    # Strict web search trigger rule: Only trigger if explicitly asked for real-time/latest info
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
        ("Gemini-1", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent", os.getenv("GEMINI_API_KEY_1"), "gemini", "query"),
        ("Mistral-1", "https://api.mistral.ai/v1/chat/completions", os.getenv("MISTRAL_API_KEY_1"), "mistral-small-latest", "bearer"),
        ("Qwen-1", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", os.getenv("QWEN_API_KEY_1"), "qwen-max", "bearer")
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
        mode = data.get("mode", "general")
        
        if not user_message and not image_data:
            raise HTTPException(status_code=400, detail="Content required")
            
        ai_reply = await call_ai_with_failover(user_message, user_id, mode, image_data)
        return {"response": ai_reply}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def home_workspace():
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NYLUVO X AI - Master Workspace</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root.dark {
                --bg-main: #0b0f19; --bg-sidebar: #030712; --bg-chat: #111827;
                --border-color: rgba(255, 255, 255, 0.08); --text-main: #f9fafb; --text-muted: #9ca3af; 
                --accent: #2563eb; --accent-hover: #1d4ed8; --hover-bg: #1f2937;
                --shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            }
            :root.light {
                --bg-main: #f3f4f6; --bg-sidebar: #ffffff; --bg-chat: #ffffff;
                --border-color: rgba(0, 0, 0, 0.1); --text-main: #111827; --text-muted: #6b7280; 
                --accent: #2563eb; --accent-hover: #1d4ed8; --hover-bg: #e5e7eb;
                --shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
            body { background: var(--bg-main); color: var(--text-main); display: flex; height: 100vh; height: 100dvh; overflow: hidden; position: relative; transition: background 0.3s; }
            
            .sidebar { width: 280px; background: var(--bg-sidebar); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding: 16px; height: 100%; z-index: 100; transition: background 0.3s; }
            .brand { font-size: 16px; font-weight: 700; color: var(--text-main); margin-bottom: 20px; display: flex; align-items: center; gap: 8px; padding: 4px 8px; letter-spacing: 0.5px; }
            .new-chat-btn { background: var(--accent); color: #fff; border: none; padding: 12px 16px; border-radius: 12px; font-weight: 600; font-size: 14px; cursor: pointer; text-align: left; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; width: 100%; transition: background 0.2s; }
            .new-chat-btn:hover { background: var(--accent-hover); }
            
            .chat-history { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; padding: 0 4px; }
            .chat-history::-webkit-scrollbar { width: 4px; }
            .chat-history::-webkit-scrollbar-thumb { background: rgba(150,150,150,0.2); border-radius: 4px; }
            .history-item { padding: 12px 14px; font-size: 13.5px; color: var(--text-muted); border-radius: 10px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s; user-select: none; }
            .history-item:hover { background: var(--hover-bg); color: var(--text-main); }
            .history-item .del-btn { opacity: 0; transition: opacity 0.2s; background: none; border: none; color: #f87171; cursor: pointer; font-size: 14px; padding: 2px 6px; }
            .history-item:hover .del-btn { opacity: 1; }

            .sidebar-footer { border-top: 1px solid var(--border-color); padding-top: 12px; display: flex; flex-direction: column; gap: 6px; }
            .footer-btn { color: var(--text-muted); font-size: 14px; padding: 12px 14px; border-radius: 10px; display: flex; align-items: center; gap: 12px; background: transparent; border: none; width: 100%; cursor: pointer; text-align: left; transition: all 0.2s; }
            .footer-btn:hover { background: var(--hover-bg); color: var(--text-main); }

            .main-container { flex: 1; display: flex; flex-direction: column; background: var(--bg-main); position: relative; height: 100%; min-width: 0; }
            .chat-header { padding: 16px 24px; border-bottom: 1px solid var(--border-color); font-weight: 600; font-size: 14px; display: flex; align-items: center; justify-content: space-between; background: var(--bg-main); z-index: 10; }
            
            .mode-selector { display: flex; gap: 8px; }
            .mode-btn { background: var(--bg-chat); border: 1px solid var(--border-color); color: var(--text-muted); padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
            .mode-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }

            .chat-messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 28px; align-items: center; scroll-behavior: smooth; }
            .chat-messages::-webkit-scrollbar { width: 6px; }
            .chat-messages::-webkit-scrollbar-thumb { background: rgba(150,150,150,0.2); border-radius: 4px; }
            
            .welcome-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 100%; max-width: 600px; margin: auto; gap: 16px; animation: fadeIn 0.4s ease; }
            .welcome-screen h1 { font-size: 28px; font-weight: 700; color: var(--text-main); }
            .welcome-screen p { color: var(--text-muted); font-size: 15px; line-height: 1.6; }
            .suggestion-chips { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; width: 100%; margin-top: 10px; }
            .chip { background: var(--bg-chat); border: 1px solid var(--border-color); padding: 14px; border-radius: 12px; font-size: 13.5px; color: var(--text-main); cursor: pointer; text-align: left; transition: border-color 0.2s; }
            .chip:hover { border-color: var(--accent); }

            .message-wrapper { width: 100%; max-width: 780px; display: flex; gap: 16px; font-size: 15px; line-height: 1.7; position: relative; animation: fadeIn 0.3s ease; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
            .message-wrapper.user { justify-content: flex-end; }
            .message-bubble { padding: 14px 18px; border-radius: 18px; max-width: 82%; word-break: break-word; box-shadow: var(--shadow); }
            .message-wrapper.user .message-bubble { background: var(--accent); color: #fff; border-top-right-radius: 4px; }
            .message-wrapper.ai .message-bubble { background: var(--bg-chat); border: 1px solid var(--border-color); color: var(--text-main); border-top-left-radius: 4px; }
            
            .typing-cursor::after { content: '▋'; display: inline-block; animation: blink 1s infinite; color: var(--accent); margin-left: 2px; font-size: 12px; vertical-align: baseline; }
            @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

            .input-container { padding: 16px 24px 28px 24px; background: var(--bg-main); display: flex; justify-content: center; }
            .input-box { width: 100%; max-width: 780px; background: var(--bg-chat); border: 1px solid var(--border-color); border-radius: 24px; display: flex; flex-direction: column; padding: 12px 16px; box-shadow: var(--shadow); transition: border-color 0.2s; }
            .input-box:focus-within { border-color: var(--accent); }
            .input-top { display: flex; align-items: flex-end; gap: 12px; }
            .input-box textarea { flex: 1; background: transparent; border: none; color: var(--text-main); font-size: 15px; resize: none; outline: none; padding: 6px; max-height: 180px; }
            .input-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
            
            .action-tools { display: flex; gap: 10px; align-items: center; }
            .icon-btn { background: transparent; border: none; cursor: pointer; color: var(--text-muted); font-size: 18px; display: flex; align-items: center; justify-content: center; transition: color 0.2s; }
            .icon-btn:hover { color: var(--accent); }
            
            .send-btn { background: var(--accent); color: #fff; border: none; width: 38px; height: 38px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; font-weight: bold; transition: background 0.2s, transform 0.1s; }
            .send-btn:hover { background: var(--accent-hover); transform: scale(1.05); }

            .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); backdrop-filter: blur(5px); display: flex; align-items: center; justify-content: center; z-index: 1000; }
            .modal-card { background: var(--bg-chat); border: 1px solid var(--border-color); padding: 32px; border-radius: 24px; width: 420px; display: flex; flex-direction: column; gap: 16px; box-shadow: 0 25px 50px rgba(0,0,0,0.6); }
            .primary-btn { background: var(--accent); color: #fff; padding: 14px; border-radius: 12px; border: none; font-weight: 600; cursor: pointer; font-size: 14px; transition: background 0.2s; }
            .primary-btn:hover { background: var(--accent-hover); }

            #imagePreviewContainer { display: none; padding: 8px 12px; gap: 8px; align-items: center; border-bottom: 1px solid var(--border-color); }
            #imagePreviewContainer img { width: 40px; height: 40px; border-radius: 8px; object-fit: cover; }
        </style>
    </head>
    <body>
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
            </div>
        </div>

        <div class="main-container">
            <div class="chat-header">
                <span id="currentChatTitle">New Workspace</span>
                <div class="mode-selector">
                    <button class="mode-btn active" id="mode-general" onclick="setMode('general')">General</button>
                    <button class="mode-btn" id="mode-student" onclick="setMode('student')">Student</button>
                    <button class="mode-btn" id="mode-professional" onclick="setMode('professional')">Professional</button>
                </div>
            </div>
            
            <div class="chat-messages" id="chatWindow"></div>

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

            let currentMode = 'general';
            function setMode(mode) {
                currentMode = mode;
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                document.getElementById('mode-' + mode).classList.add('active');
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

            let chats = JSON.parse(localStorage.getItem('chats')) || [{ id: Date.now(), title: 'New Workspace', messages: [] }];
            let activeChatId = chats[0].id;
            let currentUserId = localStorage.getItem('nyluvo_user_id') || 'user_' + Math.random().toString(36).substring(7);
            localStorage.setItem('nyluvo_user_id', currentUserId);

            function saveChats() { localStorage.setItem('chats', JSON.stringify(chats)); renderHistory(); }
            
            function renderHistory() {
                const list = document.getElementById('chatHistoryList');
                list.innerHTML = '<div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); padding: 4px 6px; font-weight: 700; letter-spacing: 0.5px;">Recent Chats</div>';
                
                chats.forEach(chat => {
                    const item = document.createElement('div');
                    item.className = 'history-item';
                    item.innerHTML = `<span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" onclick="switchChat(${chat.id})">${chat.title}</span><button class="del-btn" onclick="deleteChat(event, ${chat.id})" title="Delete Chat">✕</button>`;
                    
                    let pressTimer;
                    item.addEventListener('touchstart', function() {
                        pressTimer = setTimeout(function() {
                            if(confirm("Delete this chat session?")) { deleteChatDirect(${chat.id}); }
                        }, 800);
                    });
                    item.addEventListener('touchend', function() { clearTimeout(pressTimer); });
                    list.appendChild(item);
                });
            }

            function deleteChat(event, id) {
                event.stopPropagation();
                if(chats.length <= 1) { alert("At least one chat required."); return; }
                chats = chats.filter(c => c.id !== id);
                if(activeChatId === id) activeChatId = chats[0].id;
                saveChats(); loadActiveChat();
            }

            function deleteChatDirect(id) {
                if(chats.length <= 1) return;
                chats = chats.filter(c => c.id !== id);
                if(activeChatId === id) activeChatId = chats[0].id;
                saveChats(); loadActiveChat();
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

            function sendSuggestion(text) {
                document.getElementById('userInput').value = text;
                sendMessage();
            }

            function loadActiveChat() {
                const chat = chats.find(c => c.id === activeChatId);
                if (!chat) return;
                document.getElementById('currentChatTitle').innerText = chat.title;
                const window = document.getElementById('chatWindow');
                window.innerHTML = '';
                
                if(chat.messages.length === 0) {
                    window.innerHTML = `
                        <div class="welcome-screen">
                            <h1>Namaste! Main hoon NYLUVO X AI 😊</h1>
                            <p>Mujhe Mr. Sonu ne NYLUVO X AI Pvt. Ltd. ke antargat banaya hai. Aaj main aapki kya madad kar sakta hoon?</p>
                            <div class="suggestion-chips">
                                <div class="chip" onclick="sendSuggestion('Python mein ek simple web scraper likh kar do')">💻 Python code likhein</div>
                                <div class="chip" onclick="sendSuggestion('Quantum physics ko aasan shabdon mein samjhayein')">🎓 Quantum Physics samjhayein</div>
                                <div class="chip" onclick="sendSuggestion('Ek professional email draft karo leave ke liye')">✉️ Leave application email</div>
                                <div class="chip" onclick="sendSuggestion('Aaj ki technology trends kya hain?')">🚀 Tech trends batayein</div>
                            </div>
                        </div>
                    `;
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
                        body: JSON.stringify({ message: text, image: currentImg, user_id: currentUserId, mode: currentMode })
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

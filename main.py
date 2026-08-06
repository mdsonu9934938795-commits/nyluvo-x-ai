import os
import random
import base64
import httpx
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Nyluvo X AI - Ultimate SaaS Engine", version="11.0.0")

def get_keys(env_var_name):
    val = os.getenv(env_var_name, "")
    return [k.strip() for k in val.replace(",", " ").split() if k.strip()]

GROQ_KEYS = get_keys("GROQ_API_KEY")
GEMINI_KEYS = get_keys("GEMINI_API_KEY")
MISTRAL_KEYS = get_keys("MISTRAL_API_KEY")
CEREBRAS_KEYS = get_keys("CEREBRAS_API_KEY")
COHERE_KEYS = get_keys("COHERE_API_KEY")
TAVILY_KEYS = get_keys("TAVILY_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

class ChatPayload(BaseModel):
    prompt: str
    model: str = "turbo"
    use_web_search: bool = False
    session_id: str = "default_user"

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nyluvo X AI - Professional SaaS</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script>
            tailwind.config = {
                darkMode: 'class',
                theme: {
                    extend: {
                        colors: {
                            chatgpt: { dark: '#212121', sidebar: '#171717', hover: '#2f2f2f', border: '#2f2f2f' }
                        }
                    }
                }
            }
        </script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-[#212121] text-gray-100 font-sans h-screen flex overflow-hidden">

        <div id="sidebar" class="fixed inset-y-0 left-0 z-50 w-[260px] bg-[#171717] border-r border-[#2f2f2f] flex flex-col transition-transform duration-300 -translate-x-full md:translate-x-0 shadow-2xl">
            <div class="p-3 flex items-center justify-between">
                <button onclick="startNewChat()" class="flex-1 flex items-center gap-2 bg-transparent hover:bg-[#2f2f2f] border border-[#3f3f3f] text-white py-2.5 px-3 rounded-lg text-sm font-medium transition-all">
                    <i class="fa-solid fa-plus text-xs"></i> New chat
                </button>
                <button onclick="toggleSidebar()" class="md:hidden ml-2 text-gray-400 hover:text-white p-2"><i class="fa-solid fa-xmark"></i></button>
            </div>

            <div class="px-3 mb-2">
                <div class="relative">
                    <i class="fa-solid fa-search absolute left-3 top-2.5 text-gray-500 text-xs"></i>
                    <input type="text" id="searchChatsInput" oninput="filterChatHistory()" placeholder="Search chats..." class="w-full bg-[#212121] border border-[#3f3f3f] rounded-lg pl-8 pr-3 py-1.5 text-xs focus:outline-none focus:border-gray-400 text-gray-200">
                </div>
            </div>

            <div class="flex-1 overflow-y-auto px-3 space-y-1 mt-1" id="chatListContainer">
                <div class="text-[10px] font-semibold text-gray-500 px-3 py-1 uppercase tracking-wider">Pinned</div>
                <div id="pinnedList" class="space-y-1"></div>
                
                <div class="text-[10px] font-semibold text-gray-500 px-3 py-1 uppercase tracking-wider mt-4">Recent Library</div>
                <div id="recentList" class="space-y-1"></div>
            </div>

            <div class="p-3 border-t border-[#2f2f2f]">
                <div onclick="alert('Redirecting to Nyluvo Pro Secure Checkout...')" class="flex items-center gap-3 p-2.5 rounded-lg bg-gradient-to-r from-purple-900/40 to-indigo-900/40 border border-purple-500/30 cursor-pointer hover:border-purple-500 transition shadow-inner">
                    <div class="w-7 h-7 rounded-full bg-purple-600 flex items-center justify-center font-bold text-white text-xs"><i class="fa-solid fa-crown"></i></div>
                    <div class="flex-1 truncate">
                        <div class="text-xs font-bold text-white">Nyluvo Pro</div>
                        <div class="text-[10px] text-purple-300">Unlock Unlimited AI Models</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="flex-1 flex flex-col md:pl-[260px] h-full relative">
            <header class="h-12 border-b border-[#2f2f2f] flex items-center justify-between px-4 bg-[#212121]/90 backdrop-blur z-20">
                <div class="flex items-center gap-3">
                    <button onclick="toggleSidebar()" class="md:hidden text-gray-400 hover:text-white text-base"><i class="fa-solid fa-bars"></i></button>
                    <select id="modelSelect" class="bg-transparent text-sm font-semibold text-gray-200 focus:outline-none cursor-pointer">
                        <option value="turbo" class="bg-[#212121]">⚡ Nyluvo Turbo (Lightning Fast)</option>
                        <option value="pro" class="bg-[#212121]">✨ Nyluvo Pro (Vision & Deep Reasoning)</option>
                        <option value="human" class="bg-[#212121]">🧠 Nyluvo Conversational Engine</option>
                    </select>
                </div>
            </header>

            <div id="chatWindow" class="flex-1 overflow-y-auto px-4 py-6 space-y-6 flex flex-col items-center">
                <div class="max-w-2xl w-full text-center mt-24 space-y-3">
                    <h1 class="text-3xl font-semibold tracking-tight text-gray-200">What can I help with today?</h1>
                    <p class="text-xs text-gray-400">Powered by Nyluvo X Advanced Intelligence Infrastructure</p>
                </div>
            </div>

            <div class="p-4 bg-[#212121] flex justify-center">
                <div class="max-w-3xl w-full relative bg-[#2f2f2f] rounded-3xl p-3 shadow-2xl border border-transparent focus-within:border-gray-500 transition-all">
                    
                    <div id="imagePreviewContainer" class="hidden px-3 pb-2 flex items-center gap-2">
                        <img id="imagePreview" class="w-12 h-12 object-cover rounded-lg border border-gray-600">
                        <span id="imageName" class="text-xs text-gray-300 truncate max-w-xs"></span>
                        <button onclick="removeImage()" class="text-gray-400 hover:text-red-400 text-xs ml-auto"><i class="fa-solid fa-xmark"></i></button>
                    </div>

                    <textarea id="userInput" rows="1" placeholder="Message Nyluvo X..." class="w-full bg-transparent resize-none focus:outline-none px-3 text-sm text-gray-100 placeholder-gray-400 max-h-36"></textarea>
                    
                    <div class="flex items-center justify-between px-2 pt-2">
                        <div class="flex items-center gap-3">
                            <label class="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer hover:text-white bg-[#3f3f3f]/50 hover:bg-[#3f3f3f] px-3 py-1.5 rounded-full transition">
                                <input type="file" id="imageInput" accept="image/*" class="hidden" onchange="previewImage(event)">
                                <i class="fa-solid fa-image text-purple-400"></i> Upload Image
                            </label>

                            <label class="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer hover:text-white bg-[#3f3f3f]/50 hover:bg-[#3f3f3f] px-3 py-1.5 rounded-full transition">
                                <input type="checkbox" id="webSearchToggle" class="hidden">
                                <i class="fa-solid fa-globe text-purple-400"></i> Web Search
                            </label>
                        </div>

                        <button onclick="sendMessage()" class="bg-white text-black hover:bg-gray-200 w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition shadow-md">
                            <i class="fa-solid fa-arrow-up"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let selectedImageFile = null;
            let chatHistoryState = [
                { id: 1, title: "🚀 Nyluvo Architecture Setup", pinned: true },
                { id: 2, title: "Python FastAPI Multi-Key Engine", pinned: false }
            ];

            function renderSidebarLibrary() {
                const pinnedContainer = document.getElementById('pinnedList');
                const recentContainer = document.getElementById('recentList');
                pinnedContainer.innerHTML = '';
                recentContainer.innerHTML = '';

                chatHistoryState.forEach(chat => {
                    const itemHTML = `
                        <div class="group flex items-center justify-between px-3 py-2 rounded-lg hover:bg-[#2f2f2f] cursor-pointer text-xs text-gray-300 transition">
                            <span class="truncate flex-1" onclick="loadChat(${chat.id})">${chat.title}</span>
                            <div class="hidden group-hover:flex items-center gap-2">
                                <i onclick="togglePin(${chat.id})" class="fa-solid fa-thumbtack text-gray-400 hover:text-white" title="Pin / Unpin"></i>
                                <i onclick="deleteChat(${chat.id})" class="fa-solid fa-trash text-gray-400 hover:text-red-400" title="Delete Chat"></i>
                            </div>
                        </div>
                    `;
                    if (chat.pinned) pinnedContainer.innerHTML += itemHTML;
                    else recentContainer.innerHTML += itemHTML;
                });
            }

            function toggleSidebar() { document.getElementById('sidebar').classList.toggle('-translate-x-full'); }
            
            function startNewChat() {
                document.getElementById('chatWindow').innerHTML = `
                    <div class="max-w-2xl w-full text-center mt-24 space-y-3">
                        <h1 class="text-3xl font-semibold tracking-tight text-gray-200">What can I help with today?</h1>
                        <p class="text-xs text-gray-400">Powered by Nyluvo X Advanced Intelligence Infrastructure</p>
                    </div>
                `;
            }

            function togglePin(id) {
                const target = chatHistoryState.find(c => c.id === id);
                if (target) target.pinned = !target.pinned;
                renderSidebarLibrary();
            }

            function deleteChat(id) {
                chatHistoryState = chatHistoryState.filter(c => c.id !== id);
                renderSidebarLibrary();
            }

            function filterChatHistory() {
                const query = document.getElementById('searchChatsInput').value.toLowerCase();
                const items = document.querySelectorAll('#chatListContainer .group');
                items.forEach(item => {
                    const text = item.querySelector('span').innerText.toLowerCase();
                    item.style.display = text.includes(query) ? 'flex' : 'none';
                });
            }

            function previewImage(event) {
                const file = event.target.files[0];
                if (file) {
                    selectedImageFile = file;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('imagePreview').src = e.target.result;
                        document.getElementById('imageName').innerText = file.name;
                        document.getElementById('imagePreviewContainer').classList.remove('hidden');
                    }
                    reader.readAsDataURL(file);
                }
            }

            function removeImage() {
                selectedImageFile = null;
                document.getElementById('imageInput').value = '';
                document.getElementById('imagePreviewContainer').classList.add('hidden');
            }

            renderSidebarLibrary();

            async function sendMessage() {
                const input = document.getElementById('userInput');
                const prompt = input.value.trim();
                const model = document.getElementById('modelSelect').value;
                const useWebSearch = document.getElementById('webSearchToggle').checked;
                const chatWindow = document.getElementById('chatWindow');

                if (!prompt && !selectedImageFile) return;
                if (chatWindow.querySelector('.max-w-2xl')) chatWindow.innerHTML = '';

                let imageHtmlTag = selectedImageFile ? `<div class="mb-2"><img src="${document.getElementById('imagePreview').src}" class="max-w-xs rounded-lg"></div>` : '';
                
                chatWindow.innerHTML += `
                    <div class="w-full max-w-3xl flex justify-end">
                        <div class="bg-[#2f2f2f] text-white px-4 py-3 rounded-3xl max-w-xl text-sm leading-relaxed shadow">
                            ${imageHtmlTag}
                            ${prompt}
                        </div>
                    </div>
                `;

                // Add to library dynamically if first message
                if (chatHistoryState.length < 5 && prompt) {
                    chatHistoryState.unshift({ id: Date.now(), title: prompt.substring(0, 25) + '...', pinned: false });
                    renderSidebarLibrary();
                }

                input.value = '';
                const currentImg = selectedImageFile;
                removeImage();
                
                const loadingId = 'loading-' + Date.now();
                chatWindow.innerHTML += `
                    <div id="${loadingId}" class="w-full max-w-3xl flex justify-start">
                        <div class="text-gray-400 px-2 py-3 text-sm flex items-center gap-2">
                            <i class="fa-solid fa-circle-notch animate-spin text-purple-400"></i> Nyluvo is analyzing...
                        </div>
                    </div>
                `;
                chatWindow.scrollTop = chatWindow.scrollHeight;

                try {
                    const formData = new FormData();
                    formData.append('prompt', prompt);
                    formData.append('model', model);
                    formData.append('use_web_search', useWebSearch);
                    if (currentImg) formData.append('image', currentImg);

                    const response = await fetch('/api/chat', { method: 'POST', body: formData });
                    const data = await response.json();
                    document.getElementById(loadingId).remove();

                    chatWindow.innerHTML += `
                        <div class="w-full max-w-3xl flex justify-start">
                            <div class="text-gray-200 px-2 py-3 max-w-3xl text-sm leading-relaxed space-y-2">
                                ${marked.parse(data.reply)}
                            </div>
                        </div>
                    `;
                } catch (err) {
                    document.getElementById(loadingId).innerHTML = `<span class="text-red-400">Connection error. Please try again.</span>`;
                }
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/chat")
async def chat_endpoint(
    prompt: str = Form(""),
    model: str = Form("turbo"),
    use_web_search: bool = Form(False),
    image: UploadFile = None
):
    system_prompt = (
        "You are Nyluvo X AI, an advanced, ultra-intelligent, and friendly AI assistant. "
        "Always communicate in a natural, conversational, and direct human-like tone. "
        "Never reveal or mention underlying provider keys, technical routing, or third-party engines."
    )

    image_bytes = await image.read() if image else None
    ai_reply = ""
    success = False

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Multimodal Vision Routing via Gemini if image is provided
        if image_bytes and GEMINI_KEYS:
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            for key in random.sample(GEMINI_KEYS, len(GEMINI_KEYS)):
                try:
                    res = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={key}",
                        json={
                            "contents": [{
                                "parts": [
                                    {"text": system_prompt + "\n\nUser: " + prompt},
                                    {"inline_data": {"mime_type": image.content_type, "data": encoded_image}}
                                ]
                            }]
                        }
                    )
                    if res.status_code == 200:
                        ai_reply = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                        success = True
                        break
                except Exception:
                    continue

        # Standard Text Routing via Groq/Cerebras/Mistral/Cohere
        if not success:
            pools = [GROQ_KEYS, CEREBRAS_KEYS, COHERE_KEYS, MISTRAL_KEYS, GEMINI_KEYS]
            valid_pools = [p for p in pools if p]
            if valid_pools:
                chosen_pool = random.choice(valid_pools)
                key = random.choice(chosen_pool)
                try:
                    res = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt if prompt else "Describe this image."}
                            ],
                            "temperature": 0.7
                        }
                    )
                    if res.status_code == 200:
                        ai_reply = res.json()["choices"][0]["message"]["content"]
                        success = True
                except Exception:
                    pass

    if not success:
        ai_reply = "Hello! Main **Nyluvo X AI** hoon. Filhal high traffic hone ki wajah se aapka message process nahi ho paya, kripya dobara try karein."

    # Tavily Web Search integration
    if use_web_search and TAVILY_KEYS:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                search_res = await client.post(
                    "https://api.tavily.com/search", 
                    json={"api_key": random.choice(TAVILY_KEYS), "query": prompt, "max_results": 3}
                )
                if search_res.status_code == 200:
                    sources = "\n".join([f"- [{item['title']}]({item['url']})" for item in search_res.json().get("results", [])])
                    ai_reply += f"\n\n**🌐 Verified Sources:**\n{sources}"
        except Exception:
            pass

    return JSONResponse({"reply": ai_reply})

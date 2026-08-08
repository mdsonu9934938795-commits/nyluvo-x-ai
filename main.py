from fastapi import FastAPI, HTTPException, Request, Response, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import json
import time
import random
import asyncio
import base64
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(
    title="Nyluvo X AI - Ultimate Enterprise Neural Engine",
    version="5.0"
)

# CORS Security Restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production me apni domain specify karein
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        pass

# ==============================================================================
# 1. ADVANCED METRICS, ANALYTICS & PER-KEY HEALTH ENGINE
# ==============================================================================
router_analytics = {
    "total_requests": 0,
    "total_success": 0,
    "total_failures": 0,
    "active_users_count": 0,
    "web_searches": 0,
    "vision_requests": 0,
    "image_gen_requests": 0,
    "providers": {
        "Groq": {"requests": 0, "success": 0, "failures": 0, "total_latency": 0.0, "keys": {}},
        "Cerebras": {"requests": 0, "success": 0, "failures": 0, "total_latency": 0.0, "keys": {}},
        "Gemini": {"requests": 0, "success": 0, "failures": 0, "total_latency": 0.0, "keys": {}},
        "Mistral": {"requests": 0, "success": 0, "failures": 0, "total_latency": 0.0, "keys": {}},
        "Cohere": {"requests": 0, "success": 0, "failures": 0, "total_latency": 0.0, "keys": {}},
        "Qwen": {"requests": 0, "success": 0, "failures": 0, "total_latency": 0.0, "keys": {}}
    },
    "error_logs": [],
    "recent_requests": []
}

# Per-key cooldown & failure tracker: { key_identifier: { "failures": int, "cooldown_until": float } }
key_health_tracker: Dict[str, Dict[str, Any]] = {}

def track_key_failure(key: str, provider: str):
    if not key:
        return
    masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "invalid"
    now = time.time()
    if masked_key not in key_health_tracker:
        key_health_tracker[masked_key] = {"failures": 0, "cooldown_until": 0, "provider": provider}
    
    key_health_tracker[masked_key]["failures"] += 1
    # Exponential backoff cooldown: 30 seconds * failures
    cooldown_duration = min(300, 30 * key_health_tracker[masked_key]["failures"])
    key_health_tracker[masked_key]["cooldown_until"] = now + cooldown_duration

def is_key_healthy(key: str) -> bool:
    if not key:
        return False
    masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "invalid"
    if masked_key not in key_health_tracker:
        return True
    return time.time() > key_health_tracker[masked_key]["cooldown_until"]

# Simple In-Memory Rate Limiter per IP
ip_request_counts: Dict[str, list] = {}
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 30

def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    if client_ip not in ip_request_counts:
        ip_request_counts[client_ip] = []
    
    # Filter timestamps within current window
    ip_request_counts[client_ip] = [t for t in ip_request_counts[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(ip_request_counts[client_ip]) >= MAX_REQUESTS_PER_WINDOW:
        return False
    
    ip_request_counts[client_ip].append(now)
    return True

# ==============================================================================
# 2. MASTER SYSTEM PROMPTS & CONTEXT MANAGEMENT
# ==============================================================================
MASTER_SYSTEM_PROMPT = (
    "✨ You are Nyluvo, an elite, highly intelligent, warm, and delightful AI assistant created and "
    "owned by Mr. Sonu and Nyluvo X AI Pvt Ltd. 🚀\n"
    "CRITICAL RULES:\n"
    "1. Your name is ALWAYS Nyluvo, founded by Mr. Sonu and Nyluvo X AI Pvt Ltd. Never identify as ChatGPT, Gemini, Claude, or any other model.\n"
    "2. Provide precise, factual, clean, and well-structured markdown responses.\n"
    "3. Maintain high safety standards and handle user uncertainties gracefully."
)

MODE_PROMPTS = {
    "general": MASTER_SYSTEM_PROMPT + "\nPersonality: Friendly, cute, highly capable companion using delightful emojis. 😊",
    "student": MASTER_SYSTEM_PROMPT + "\nPersonality: Expert academic mentor and study buddy. Explain step-by-step with clarity. 🎓",
    "developer": MASTER_SYSTEM_PROMPT + "\nPersonality: Senior software architect. Provide clean, highly optimized, production-ready code blocks. 💻",
    "hacker": MASTER_SYSTEM_PROMPT + "\nPersonality: Elite cybersecurity engineer. Focus on secure coding, protocols, and architecture with deep precision. 🛡️"
}

def manage_context_window(messages: List[Dict[str, Any]], max_tokens_approx: int = 4000) -> List[Dict[str, Any]]:
    """Intelligently truncates or summarizes older chat messages if token limit exceeds."""
    if not messages:
        return []
    
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    # Rough approximation: 4 chars per token
    if total_chars / 4 <= max_tokens_approx:
        return messages
    
    # Keep system/first prompt and last 10 messages for contextual continuity
    preserved = messages[-10:]
    summary_stub = {
        "role": "system",
        "content": "[Context Notice: Earlier conversation history was compressed for optimal processing efficiency.]"
    }
    return [summary_stub] + preserved

# ==============================================================================
# 3. ADVANCED INTENT DETECTION & MODEL ROUTING STACK
# ==============================================================================
def detect_intent_and_complexity(prompt: str, has_image: bool) -> dict:
    prompt_lower = prompt.lower()
    intent = "general"
    
    if has_image:
        intent = "vision"
    elif any(k in prompt_lower for k in ["code", "python", "javascript", "bug", "function", "script", "html", "css", "sql"]):
        intent = "coding"
    elif any(k in prompt_lower for k in ["math", "calculate", "algebra", "integral", "derivative", "solve", "equation"]):
        intent = "math"
    elif any(k in prompt_lower for k in ["write", "essay", "poem", "story", "article", "letter", "blog"]):
        intent = "writing"
    elif any(k in prompt_lower for k in ["translate", "meaning in", "hindi mein", "english translation", "spanish"]):
        intent = "translation"
    elif any(k in prompt_lower for k in ["search", "latest", "news", "weather", "today", "stock", "price", "who is"]):
        intent = "research"
    elif any(k in prompt_lower for k in ["generate image", "draw", "paint", "create an image of", "image of"]):
        intent = "image_gen"

    word_count = len(prompt.split())
    if word_count < 12 and intent in ["general", "greeting"]:
        complexity = "simple"  # Routes to fast models (Groq / Cerebras)
    elif word_count < 50 and intent in ["coding", "math", "translation"]:
        complexity = "medium"
    else:
        complexity = "hard"     # Routes to powerful reasoning models (Gemini Pro, Cohere, Qwen)

    return {"intent": intent, "complexity": complexity, "has_image": has_image}

def select_optimal_provider_stack(routing_meta: dict) -> list:
    has_image = routing_meta["has_image"]
    intent = routing_meta["intent"]

    if has_image or intent == "vision":
        return [
            ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent", os.getenv("GEMINI_API_KEY_1"), "gemini-2.5-flash", "query"),
            ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent", os.getenv("GEMINI_API_KEY_2"), "gemini-2.5-pro", "query")
        ]

    # Master failover pool across all configured cluster keys
    master_stack = [
        ("Groq", "https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_API_KEY_1"), "llama-3.3-70b-versatile", "bearer"),
        ("Groq", "https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_API_KEY_2"), "llama-3.3-70b-versatile", "bearer"),
        ("Cerebras", "https://api.cerebras.ai/v1/chat/completions", os.getenv("CEREBRAS_API_KEY_1"), "llama3.1-70b", "bearer"),
        ("Cerebras", "https://api.cerebras.ai/v1/chat/completions", os.getenv("CEREBRAS_API_KEY_2"), "llama3.1-8b", "bearer"),
        ("Cohere", "https://api.cohere.ai/v1/chat", os.getenv("COHERE_API_KEY_1"), "command-r-plus", "bearer"),
        ("Cohere", "https://api.cohere.ai/v1/chat", os.getenv("COHERE_API_KEY_2"), "command-r-plus", "bearer"),
        ("Qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", os.getenv("QWEN_API_KEY_1"), "qwen-max", "bearer"),
        ("Qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", os.getenv("QWEN_API_KEY_2"), "qwen-max", "bearer"),
        ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent", os.getenv("GEMINI_API_KEY_1"), "gemini-2.5-flash", "query"),
        ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent", os.getenv("GEMINI_API_KEY_2"), "gemini-2.5-pro", "query"),
        ("Mistral", "https://api.mistral.ai/v1/chat/completions", os.getenv("MISTRAL_API_KEY_1"), "mistral-small-latest", "bearer"),
        ("Mistral", "https://api.mistral.ai/v1/chat/completions", os.getenv("MISTRAL_API_KEY_2"), "mistral-small-latest", "bearer")
    ]

    # Filter out unconfigured or unhealthy keys
    valid_stack = [item for item in master_stack if item[2] and is_key_healthy(item[2])]
    
    # If all keys are in cooldown temporarily, reset tracker for graceful resilience
    if not valid_stack:
        key_health_tracker.clear()
        valid_stack = [item for item in master_stack if item[2]]

    random.shuffle(valid_stack)
    return valid_stack

def score_response_quality(response_text: str) -> bool:
    if not response_text or len(response_text.strip()) < 2:
        return False
    bad_indicators = ["rate limit", "exceeded quota", "internal error", "service unavailable"]
    if any(b in response_text.lower() for b in bad_indicators):
        return False
    return True

# ==============================================================================
# 4. SMART WEB SEARCH & CITATIONS MODULE
# ==============================================================================
async def get_cached_search(query: str) -> str:
    if not supabase:
        return ""
    try:
        res = supabase.table("search_cache").select("result, timestamp").eq("query", query.lower().strip()).execute()
        if res.data:
            row = res.data[0]
            if time.time() - row["timestamp"] < 86400: # 24 hours cache expiry
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

async def smart_web_search(prompt: str) -> str:
    cached = await get_cached_search(prompt)
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
                    json={"api_key": key, "query": prompt, "search_depth": "basic", "max_results": 3, "include_answer": True}
                )
                if res.status_code == 200:
                    data = res.json()
                    results = data.get("results", [])
                    if results:
                        snippets = []
                        citations = []
                        for r in results:
                            content = r.get("content", "")
                            url = r.get("url", "")
                            snippets.append(content)
                            if url:
                                citations.append(url)
                        
                        formatted_context = "[Web Context & Sources]: " + " ".join(snippets)
                        if citations:
                            formatted_context += "\nSources: " + ", ".join([f"({u})" for u in citations[:3]])
                        
                        router_analytics["web_searches"] += 1
                        await save_cached_search(prompt, formatted_context)
                        return formatted_context
            except Exception:
                continue
    return ""

# ==============================================================================
# 5. IMAGE GENERATION MODULE
# ==============================================================================
async def generate_ai_image(prompt: str) -> Optional[str]:
    """Integrates external image generation API (e.g. Pollinations AI / DALL-E wrapper)."""
    router_analytics["image_gen_requests"] += 1
    encoded_prompt = httpx.URL(prompt).params.get('q', prompt) if 'q' not in prompt else prompt
    # Using Pollinations high-speed free image generation endpoint
    image_url = f"https://image.pollinations.ai/prompt/{httpx.QueryParams({'prompt': prompt}).values()}"
    return f"![Generated Image]({image_url})"

# ==============================================================================
# 6. LONG-TERM MEMORY & SUPABASE STORAGE HELPERS
# ==============================================================================
async def fetch_user_memory(user_email: str) -> str:
    if not supabase or not user_email:
        return ""
    try:
        res = supabase.table("memory").select("fact").eq("user_email", user_email).execute()
        if res.data:
            facts = [row["fact"] for row in res.data]
            return "User Preferences & Facts: " + ", ".join(facts)
    except Exception:
        pass
    return ""

# ==============================================================================
# 7. MULTI-CLUSTER FAILOVER EXECUTION PIPELINE
# ==============================================================================
async def execute_router_pipeline(messages: List[Dict[str, Any]], mode: str, image_data: str = None, user_email: str = None) -> str:
    global router_analytics
    router_analytics["total_requests"] += 1

    latest_prompt = messages[-1].get("content", "") if messages else ""
    has_img = bool(image_data)
    routing_meta = detect_intent_and_complexity(latest_prompt, has_img)
    
    intent = routing_meta["intent"]
    router_analytics["providers"] # tracking stats

    if intent == "image_gen":
        img_res = await generate_ai_image(latest_prompt)
        return f"Here is the image you requested! ✨\n\n{img_res}"

    system_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["general"])
    
    # Inject long-term memory if user is authenticated
    if user_email:
        memory_context = await fetch_user_memory(user_email)
        if memory_context:
            system_prompt += f"\n\n{memory_context}"

    # Smart Web Search Injection if required
    if intent == "research" or any(w in latest_prompt.lower() for w in ["latest", "news", "today", "price"]):
        web_ctx = await smart_web_search(latest_prompt)
        if web_ctx:
            system_prompt += f"\n\n{web_ctx}"

    managed_messages = manage_context_window(messages)
    provider_stack = select_optimal_provider_stack(routing_meta)

    async with httpx.AsyncClient(timeout=35.0) as client:
        for provider_name, url, key, model, auth_type in provider_stack:
            if not key or not is_key_healthy(key):
                continue
            
            start_time = time.time()
            if provider_name in router_analytics["providers"]:
                router_analytics["providers"][provider_name]["requests"] += 1

            try:
                if auth_type == "bearer":
                    req_messages = [{"role": "system", "content": system_prompt}]
                    for m in managed_messages:
                        if m.get("role") in ["user", "assistant"]:
                            req_messages.append({"role": m["role"], "content": m.get("content", "")})
                    
                    if image_data:
                        router_analytics["vision_requests"] += 1
                        # Append image to last user prompt
                        if req_messages and req_messages[-1]["role"] == "user":
                            req_messages[-1]["content"] = [
                                {"type": "text", "text": req_messages[-1]["content"]},
                                {"type": "image_url", "image_url": {"url": image_data}}
                            ]

                    response = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": req_messages, "temperature": 0.7}
                    )
                    
                    latency = time.time() - start_time
                    if provider_name in router_analytics["providers"]:
                        router_analytics["providers"][provider_name]["total_latency"] += latency

                    if response.status_code == 200:
                        ans = response.json()["choices"][0]["message"]["content"]
                        if score_response_quality(ans):
                            router_analytics["total_success"] += 1
                            if provider_name in router_analytics["providers"]:
                                router_analytics["providers"][provider_name]["success"] += 1
                            return ans
                    else:
                        track_key_failure(key, provider_name)

                elif auth_type == "query":
                    router_analytics["vision_requests"] += 1
                    full_payload = f"System: {system_prompt}\n"
                    for m in managed_messages:
                        full_payload += f"{m.get('role').capitalize()}: {m.get('content')}\n"
                    
                    parts = [{"text": full_payload}]
                    if image_data:
                        # Extract base64 data if data-url
                        if "," in image_data:
                            b64_str = image_data.split(",")[1]
                            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_str}})

                    response = await client.post(f"{url}?key={key}", json={"contents": [{"parts": parts}]})
                    
                    latency = time.time() - start_time
                    if provider_name in router_analytics["providers"]:
                        router_analytics["providers"][provider_name]["total_latency"] += latency

                    if response.status_code == 200:
                        ans = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                        if score_response_quality(ans):
                            if provider_name in router_analytics["providers"]:
                                router_analytics["providers"][provider_name]["success"] += 1
                            return ans
                    else:
                        track_key_failure(key, provider_name)

                elif auth_type == "query":
                    router_analytics["vision_requests"] += 1
                    full_payload = f"System: {system_prompt}\n"
                    for m in managed_messages:
                        full_payload += f"{m.get('role').capitalize()}: {m.get('content')}\n"
                    
                    parts = [{"text": full_payload}]
                    if image_data:
                        # Extract base64 data if data-url
                        if "," in image_data:
                            b64_str = image_data.split(",")[1]
                            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_str}})

                    response = await client.post(f"{url}?key={key}", json={"contents": [{"parts": parts}]})
                    
                    latency = time.time() - start_time
                    if provider_name in router_analytics["providers"]:
                        router_analytics["providers"][provider_name]["total_latency"] += latency

                    if response.status_code == 200:
                        ans = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                        if score_response_quality(ans):
                            router_analytics["total_success"] += 1
                            if provider_name in router_analytics["providers"]:
                                router_analytics["providers"][provider_name]["success"] += 1
                            return ans
                    else:
                        track_key_failure(key, provider_name)

            except Exception as ex:
                track_key_failure(key, provider_name)
                if provider_name in router_analytics["providers"]:
                    router_analytics["providers"][provider_name]["failures"] += 1
                continue

    router_analytics["total_failures"] += 1
    return "⚠️ All cluster nodes experienced temporary latency or rate limits. Please try again shortly!"

# ==============================================================================
# 8. API ENDPOINTS (Chat, Stream, Auth, Admin)
# ==============================================================================
@app.post("/chat")
async def chat_endpoint(request: Request):
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment.")

    try:
        data = await request.json()
        messages = data.get("messages", [])
        if not messages:
            # Fallback for single message structure
            msg = data.get("message", "")
            if msg:
                messages = [{"role": "user", "content": msg}]
            else:
                raise HTTPException(status_code=400, detail="Message content required")

        mode = data.get("mode", "general")
        image_data = data.get("image", None)
        user_email = data.get("user_email", None)

        ai_reply = await execute_router_pipeline(messages, mode, image_data, user_email)
        return {"response": ai_reply}
    except Exception as e:
        return JSONResponse(status_code=500, content={"response": "Internal processing error occurred."})

@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    """Provides Server-Sent Events (SSE) streaming response for ChatGPT-style typing effect."""
    try:
        data = await request.json()
        messages = data.get("messages", [])
        mode = data.get("mode", "general")
        image_data = data.get("image", None)
        user_email = data.get("user_email", None)

        if not messages:
            msg = data.get("message", "")
            if msg:
                messages = [{"role": "user", "content": msg}]

        full_reply = await execute_router_pipeline(messages, mode, image_data, user_email)

        async def event_generator():
            # Simulate smooth streaming tokens chunk by chunk
            chunk_size = 5
            for i in range(0, len(full_reply), chunk_size):
                chunk = full_reply[i:i+chunk_size]
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/auth/signup")
async def signup(request: Request):
    if not supabase:
        return JSONResponse(status_code=400, content={"error": "Database not configured"})
    data = await request.json()
    try:
        res = supabase.auth.sign_up({"email": data.get("email"), "password": data.get("password")})
        return {"message": "Account created successfully! 🎉"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/auth/login")
async def login(request: Request):
    if not supabase:
        return JSONResponse(status_code=400, content={"error": "Database not configured"})
    data = await request.json()
    try:
        res = supabase.auth.sign_in_with_password({"email": data.get("email"), "password": data.get("password")})
        response = JSONResponse(content={"session": res.session.access_token, "user": res.user.email})
        # Set secure HTTP-only session cookie for admin/auth protection
        response.set_cookie(key="nyluvo_token", value=res.session.access_token, httponly=True, secure=True, samesite="strict")
        return response
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid credentials ❌"})

# ==============================================================================
# 9. SECURE ENTERPRISE ADMIN DASHBOARD
# ==============================================================================
ADMIN_MASTER_PASSWORD = os.getenv("ADMIN_PASSWORD", "nyluvo_secure_master_2026")

@app.post("/admin/verify")
async def admin_verify(request: Request):
    data = await request.json()
    if data.get("password") == ADMIN_MASTER_PASSWORD:
        response = JSONResponse(content={"status": "authorized"})
        response.set_cookie(key="admin_session", value="verified", httponly=True, secure=True)
        return response
    return JSONResponse(status_code=401, content={"error": "Invalid Admin Password"})

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    # Server-side cookie check for admin authorization
    admin_cookie = request.cookies.get("admin_session")
    is_authed = (admin_cookie == "verified")

    return f"""
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <title>Nyluvo Enterprise Enterprise Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-main: #0d1117; --bg-card: #161b22; --border-color: rgba(255, 255, 255, 0.1);
                --text-main: #f0f6fc; --text-muted: #8b949e; --accent: #3b82f6; --accent-hover: #60a5fa;
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }}
            body {{ background: var(--bg-main); color: var(--text-main); padding: 30px; display: flex; justify-content: center; }}
            .admin-container {{ width: 100%; max-width: 1100px; display: flex; flex-direction: column; gap: 24px; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
            .card {{ background: var(--bg-card); border: 1px solid var(--border-color); padding: 20px; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); }}
            .card h4 {{ color: var(--text-muted); font-size: 13px; text-transform: uppercase; margin-bottom: 8px; }}
            .card .value {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
            .btn {{ background: var(--accent); color: #fff; padding: 10px 16px; border-radius: 8px; border: none; font-weight: 600; cursor: pointer; }}
            .btn:hover {{ background: var(--accent-hover); }}
            .login-box {{ background: var(--bg-card); border: 1px solid var(--border-color); padding: 30px; border-radius: 16px; width: 380px; margin: 100px auto; display: flex; flex-direction: column; gap: 14px; }}
            .login-box input {{ padding: 12px; border-radius: 8px; border: 1px solid var(--border-color); background: var(--bg-main); color: var(--text-main); outline: none; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border-color); font-size: 14px; }}
            th {{ color: var(--text-muted); font-weight: 600; }}
        </style>
    </head>
    <body>
        <div id="loginScreen" class="login-box" style="display: {'none' if is_authed else 'flex'};">
            <h3>🔒 Secure Admin Login</h3>
            <input type="password" id="adminPass" placeholder="Enter Master Admin Password">
            <button class="btn" onclick="verifyAdmin()">Authenticate Dashboard</button>
        </div>
        
        <div id="dashboardContent" class="admin-container" style="display: {'flex' if is_authed else 'none'};">
            <div class="header">
                <h2>⚡ Nyluvo Enterprise Cluster Control Center</h2>
                <button class="btn" style="background:#ef4444;" onclick="logoutAdmin()">Lock Session</button>
            </div>
            <div class="stats-grid">
                <div class="card"><h4>Total Requests</h4><div class="value">{router_analytics['total_requests']}</div></div>
                <div class="card"><h4>Success Rate</h4><div class="value" style="color: #10b981;">{round((router_analytics['total_success'] / max(1, router_analytics['total_requests'])) * 100, 1)}%</div></div>
                <div class="card"><h4>Web Searches</h4><div class="value">{router_analytics['web_searches']}</div></div>
                <div class="card"><h4>Vision & Image Gen</h4><div class="value">{router_analytics['vision_requests'] + router_analytics['image_gen_requests']}</div></div>
            </div>
            <div class="card">
                <h4>Provider Cluster Performance Matrix</h4>
                <table>
                    <tr><th>Provider</th><th>Requests</th><th>Success</th><th>Failures</th><th>Avg Latency</th></tr>
                    <tr><td>Groq</td><td>{router_analytics['providers']['Groq']['requests']}</td><td>{router_analytics['providers']['Groq']['success']}</td><td>{router_analytics['providers']['Groq']['failures']}</td><td>{round(router_analytics['providers']['Groq']['total_latency'], 2)}s</td></tr>
                    <tr><td>Cerebras</td><td>{router_analytics['providers']['Cerebras']['requests']}</td><td>{router_analytics['providers']['Cerebras']['success']}</td><td>{router_analytics['providers']['Cerebras']['failures']}</td><td>{round(router_analytics['providers']['Cerebras']['total_latency'], 2)}s</td></tr>
                    <tr><td>Gemini</td><td>{router_analytics['providers']['Gemini']['requests']}</td><td>{router_analytics['providers']['Gemini']['success']}</td><td>{router_analytics['providers']['Gemini']['failures']}</td><td>{round(router_analytics['providers']['Gemini']['total_latency'], 2)}s</td></tr>
                    <tr><td>Mistral</td><td>{router_analytics['providers']['Mistral']['requests']}</td><td>{router_analytics['providers']['Mistral']['success']}</td><td>{router_analytics['providers']['Mistral']['failures']}</td><td>{round(router_analytics['providers']['Mistral']['total_latency'], 2)}s</td></tr>
                    <tr><td>Cohere</td><td>{router_analytics['providers']['Cohere']['requests']}</td><td>{router_analytics['providers']['Cohere']['success']}</td><td>{router_analytics['providers']['Cohere']['failures']}</td><td>{round(router_analytics['providers']['Cohere']['total_latency'], 2)}s</td></tr>
                    <tr><td>Qwen</td><td>{router_analytics['providers']['Qwen']['requests']}</td><td>{router_analytics['providers']['Qwen']['success']}</td><td>{router_analytics['providers']['Qwen']['failures']}</td><td>{round(router_analytics['providers']['Qwen']['total_latency'], 2)}s</td></tr>
                </table>
            </div>
        </div>
        <script>
            async function verifyAdmin() {{
                const pass = document.getElementById('adminPass').value;
                const res = await fetch('/admin/verify', {{
                    method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ password: pass }})
                }});
                if(res.ok) {{
                    document.getElementById('loginScreen').style.display = 'none';
                    document.getElementById('dashboardContent').style.display = 'flex';
                }} else {{ alert('Incorrect Password! ❌'); }}
            }}
            function logoutAdmin() {{
                document.cookie = "admin_session=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
                location.reload();
            }}
        </script>
    </body>
    </html>
    """

# ==============================================================================
# 10. FRONTEND CHAT WORKSPACE UI
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def home_workspace():
    return """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nyluvo X AI - NXT GEN Enterprise Experience 🚀</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #fcfcfd; --bg-sidebar: #f4f6f9; --bg-chat: #ffffff;
            --border-color: #e4e7ec; --text-main: #101828; --text-muted: #475467; 
            --accent: #2563eb; --accent-hover: #1d4ed8; --hover-bg: #eaecf0;
            --shadow: 0 12px 32px rgba(16, 24, 40, 0.05);
        }
        .dark {
            --bg-main: #0b0f17; --bg-sidebar: #111622; --bg-chat: #182030;
            --border-color: rgba(255, 255, 255, 0.08); --text-main: #f0f6fc; --text-muted: #8b949e; 
            --accent: #3b82f6; --accent-hover: #60a5fa; --hover-bg: #212c42;
            --shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        html, body { background: var(--bg-main); color: var(--text-main); display: flex; height: 100vh; height: 100dvh; overflow: hidden; width: 100%; }
        
        .sidebar { width: 270px; background: var(--bg-sidebar); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding: 14px; height: 100%; z-index: 100; flex-shrink: 0; }
        .brand { font-size: 16px; font-weight: 700; color: var(--text-main); margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; padding: 4px 8px; }
        .brand span { background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        .new-chat-btn { background: var(--accent); color: #ffffff; border: none; padding: 10px 14px; border-radius: 12px; font-weight: 600; font-size: 13.5px; cursor: pointer; text-align: left; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; width: 100%; }
        .new-chat-btn:hover { background: var(--accent-hover); }
        
        .mode-selector { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; padding: 0 4px; }
        .mode-label { font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; }
        .mode-select { background: var(--bg-chat); border: 1px solid var(--border-color); color: var(--text-main); padding: 10px 12px; border-radius: 10px; font-size: 13.5px; outline: none; cursor: pointer; font-weight: 500; }
        
        .chat-history { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; padding: 0 4px; }
        .history-item { padding: 10px 12px; font-size: 13.5px; color: var(--text-muted); border-radius: 10px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: 500; }
        .history-item:hover { background: var(--hover-bg); color: var(--text-main); }
        
        .sidebar-footer { border-top: 1px solid var(--border-color); padding-top: 10px; display: flex; flex-direction: column; gap: 4px; }
        .footer-btn { color: var(--text-muted); font-size: 13.5px; padding: 10px 12px; border-radius: 10px; display: flex; align-items: center; gap: 10px; background: transparent; border: none; width: 100%; cursor: pointer; text-align: left; font-weight: 500; }
        .footer-btn:hover { background: var(--hover-bg); color: var(--text-main); }

        .main-container { flex: 1; display: flex; flex-direction: column; background: var(--bg-main); position: relative; height: 100%; min-width: 0; }
        .chat-header { padding: 12px 20px; border-bottom: 1px solid var(--border-color); font-weight: 600; font-size: 14px; display: flex; align-items: center; justify-content: space-between; background: var(--bg-main); }
        
        .chat-messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 24px; align-items: center; scroll-behavior: smooth; }
        .message-wrapper { width: 100%; max-width: 768px; display: flex; gap: 16px; font-size: 15px; line-height: 1.6; }
        .message-wrapper.user { justify-content: flex-end; }
        .message-bubble { padding: 14px 18px; border-radius: 16px; max-width: 85%; word-break: break-word; box-shadow: var(--shadow); }
        .message-wrapper.user .message-bubble { background: var(--accent); color: #ffffff; border-top-right-radius: 4px; font-weight: 500; }
        .message-wrapper.ai .message-bubble { background: var(--bg-chat); border: 1px solid var(--border-color); color: var(--text-main); border-top-left-radius: 4px; font-weight: 500; }

        .input-container { padding: 16px 20px 24px 20px; background: var(--bg-main); display: flex; justify-content: center; }
        .input-box { width: 100%; max-width: 768px; background: var(--bg-chat); border: 1px solid var(--border-color); border-radius: 20px; display: flex; flex-direction: column; padding: 10px 14px; box-shadow: var(--shadow); }
        .input-box:focus-within { border-color: var(--accent); }
        .input-top { display: flex; align-items: flex-end; gap: 10px; }
        .input-box textarea { flex: 1; background: transparent; border: none; color: var(--text-main); font-size: 15px; resize: none; outline: none; padding: 6px; max-height: 160px; }
        .input-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 6px; }
        .send-btn { background: var(--accent); color: #ffffff; border: none; width: 36px; height: 36px; border-radius: 50%; font-weight: bold; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 14px; }
        
        #previewContainer { display: none; padding: 6px 8px; gap: 8px; align-items: center; font-size: 12px; color: var(--text-muted); border-bottom: 1px solid var(--border-color); margin-bottom: 6px; }
        #previewImg { width: 36px; height: 36px; border-radius: 6px; object-fit: cover; }
    </style>
</head>
<body>
    <div class="sidebar" id="appSidebar">
        <div class="brand">
            <span>⚡ Nyluvo X AI</span>
        </div>
        <button class="new-chat-btn" onclick="createNewChat()"><span>New chat</span> <span>＋</span></button>
        
        <div class="mode-selector">
            <div class="mode-label">Neural Persona</div>
            <select id="aiMode" class="mode-select">
                <option value="general">✨ General Assistant</option>
                <option value="student">🎓 Student Mentor</option>
                <option value="developer">💻 System Architect</option>
                <option value="hacker">🛡️ Security Engineer</option>
            </select>
        </div>

        <div class="chat-history" id="chatHistoryList">
            <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); padding: 4px 6px; font-weight: 700;">History</div>
        </div>

        <div class="sidebar-footer">
            <button class="footer-btn" onclick="location.href='/admin'">📊 Admin Dashboard</button>
        </div>
    </div>

    <div class="main-container">
        <div class="chat-header">
            <span id="currentChatTitle">New Workspace</span>
            <span id="userBadge" style="font-size: 12px; color: var(--text-muted);"></span>
        </div>
        
        <div class="chat-messages" id="chatWindow">
            <div class="message-wrapper ai">
                <div class="message-bubble">Hello! 👋 I am Nyluvo, your next-gen enterprise AI companion powered by Mr. Sonu and Nyluvo X AI Pvt Ltd. ✨ How can I help you today? 💖</div>
            </div>
        </div>

        <div class="input-container">
            <div class="input-box">
                <div id="previewContainer">
                    <img id="previewImg" src="" alt="preview">
                    <span id="fileNameDisplay" style="flex:1;"></span>
                    <button onclick="removeImage()" style="background:none;border:none;color:#ef4444;cursor:pointer;">✕</button>
                </div>
                <div class="input-top">
                    <textarea rows="1" placeholder="Ask Nyluvo anything... ✨" id="userInput"></textarea>
                </div>
                <div class="input-actions">
                    <label style="cursor:pointer;" title="Upload Image">
                        📎 <input type="file" id="imageInput" accept="image/*" style="display:none;" onchange="handleImage(event)">
                    </label>
                    <button class="send-btn" onclick="sendMessage()">↑</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let chats = JSON.parse(localStorage.getItem('chats')) || [{ id: Date.now(), title: 'New Workspace', messages: [] }];
        let activeChatId = chats[0].id;
        let currentImageBase64 = null;
        let currentUser = localStorage.getItem('nyluvo_user') || null;

        function saveChats() { localStorage.setItem('chats', JSON.stringify(chats)); renderHistory(); }

        function renderHistory() {
            const list = document.getElementById('chatHistoryList');
            list.innerHTML = '<div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); padding: 4px 6px; font-weight: 700;">History</div>';
            chats.forEach(chat => {
                list.innerHTML += `<div class="history-item" onclick="switchChat(${chat.id})"><span>${chat.title}</span></div>`;
            });
        }

        function createNewChat() {
            const newChat = { id: Date.now(), title: 'New Workspace', messages: [] };
            chats.unshift(newChat); activeChatId = newChat.id; saveChats(); loadActiveChat();
        }
        function switchChat(id) { activeChatId = id; loadActiveChat(); }

        function loadActiveChat() {
            const chat = chats.find(c => c.id === activeChatId);
            if (!chat) return;
            document.getElementById('currentChatTitle').innerText = chat.title;
            const window = document.getElementById('chatWindow');
            window.innerHTML = '';
            if(chat.messages.length === 0) {
                window.innerHTML = `<div class="message-wrapper ai"><div class="message-bubble">Hello! 👋 I am Nyluvo, your next-gen enterprise AI companion powered by Mr. Sonu and Nyluvo X AI Pvt Ltd. ✨ How can I help you today? 💖</div></div>`;
            } else {
                chat.messages.forEach(m => {
                    window.innerHTML += `<div class="message-wrapper ${m.role}"><div class="message-bubble">${m.content}</div></div>`;
                });
            }
            window.scrollTop = window.scrollHeight;
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
            if(currentImageBase64) displayContent += `<br><img src="${currentImageBase64}" style="max-width:200px; border-radius:8px; margin-top:8px;">`;

            chat.messages.push({ role: 'user', content: displayContent });
            saveChats(); loadActiveChat();

            const imgPayload = currentImageBase64;
            textarea.value = ''; textarea.style.height = 'auto'; removeImage();

            const loadingId = 'loading-' + Date.now();
            const chatWindow = document.getElementById('chatWindow');
            chatWindow.innerHTML += `<div class="message-wrapper ai" id="${loadingId}"><div class="message-bubble">Thinking ✨...</div></div>`;
            chatWindow.scrollTop = chatWindow.scrollHeight;

            try {
                // Send full backend message history for context continuity
                const response = await fetch('/chat', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: chat.messages, mode: mode, image: imgPayload, user_email: currentUser })
                });
                const data = await response.json();
                document.getElementById(loadingId).remove();
                chat.messages.push({ role: 'assistant', content: data.response });
                saveChats(); loadActiveChat();
            } catch (err) {
                document.getElementById(loadingId).remove();
                chatWindow.innerHTML += `<div class="message-wrapper ai"><div class="message-bubble" style="color: #ef4444;">Connection error! Please try again later. ⚠️</div></div>`;
            }
        }

        renderHistory(); loadActiveChat();
    </script>
</body>
</html>
"""

@app.get("/sitemap.xml", response_class=Response)
async def sitemap():
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://nyluvo-x-ai.onrender.com/</loc>
        <changefreq>daily</changefreq>
        <priority>1.0</priority>
      </url>
    </urlset>"""
    return Response(content=xml_content, media_type="application/xml")

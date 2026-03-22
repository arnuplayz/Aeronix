"""
AERONIX v5
==========
- Groq LLaMA 3.3 70B   → fast AI (gaming, video prompts, science, code)
- Pollinations.ai       → real image generation (free, no key ever)
- Pillow + numpy        → image upscaling
- edge-tts              → voice output
- Flask + ngrok         → local + public server
- Drag & drop, mic, streaming, memory
"""

import asyncio, base64, datetime, glob, io, json, math
import os, re, subprocess, tempfile, threading, time, webbrowser

import edge_tts
import numpy as np
import playsound
import requests
import speech_recognition as sr
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS
from groq import Groq
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

# ================================================================
#  CONFIG  ← only edit these two lines
# ================================================================

GROQ_API_KEY  = "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"    # groq.com → API Keys
NGROK_TOKEN   = "3BHcTBSzIVrCsiFDD9BTtLvGmI2_66QECNBuHeJ7d1eo6dAc8" # dashboard.ngrok.com → Authtoken

VOICE         = "en-US-GuyNeural"
PORT          = 5000
MEMORY_FILE   = "memory.json"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================================================================
#  FLASK + GROQ
# ================================================================

app    = Flask(__name__)
CORS(app)
client = Groq(api_key=GROQ_API_KEY)

# ================================================================
#  MEMORY
# ================================================================

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

memory = load_memory()

# ================================================================
#  VOICE — speaks first sentence immediately, skips code/markdown
# ================================================================

def clean_for_speech(text):
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'^\s*[-•*]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 380:
        text = text[:360].rsplit(' ', 1)[0] + ". Check the chat for full details."
    return text

async def _speak_async(text):
    cleaned = clean_for_speech(text)
    if not cleaned:
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        fname = f.name
    await edge_tts.Communicate(cleaned, VOICE, rate="+0%").save(fname)
    playsound.playsound(fname)
    try:
        os.remove(fname)
    except:
        pass

def speak(text):
    print(f"Aeronix: {text[:80]}")
    asyncio.run(_speak_async(text))

def speak_bg(text):
    threading.Thread(
        target=lambda: asyncio.run(_speak_async(text)), daemon=True
    ).start()

# ================================================================
#  MICROPHONE
# ================================================================

recognizer = sr.Recognizer()
recognizer.energy_threshold = 250
recognizer.pause_threshold  = 0.6

# ================================================================
#  IMAGE GENERATION — Pollinations.ai (free, no API key ever)
#  Runs FLUX model internally, returns real photos/art
# ================================================================

def enhance_image_prompt(prompt):
    """Auto-boost prompt quality for better results."""
    p = prompt.lower()
    extras = []
    if not any(w in p for w in ["4k","hd","high quality","detailed","realistic","sharp"]):
        extras.append("high quality, detailed")
    if not any(w in p for w in ["photo","illustration","digital art","painting","cartoon","anime"]):
        extras.append("digital art")
    if not any(w in p for w in ["dark","horror","gloomy","black"]):
        extras.append("vibrant colors")
    return f"{prompt}, {', '.join(extras)}" if extras else prompt

def generate_placeholder(prompt):
    """Styled fallback image when all APIs fail."""
    img  = Image.new("RGB", (512, 512), color=(6, 10, 18))
    draw = ImageDraw.Draw(img)
    for x in range(0, 512, 44):
        draw.line([(x,0),(x,512)], fill=(0,50,70), width=1)
    for y in range(0, 512, 44):
        draw.line([(0,y),(512,y)], fill=(0,50,70), width=1)
    draw.text((20, 210), "[Image generation offline]", fill=(0,200,220))
    draw.text((20, 240), "Check internet connection", fill=(0,150,180))
    draw.text((20, 265), prompt[:45], fill=(0,120,150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def generate_image(prompt, width=512, height=512):
    """
    Generate real image via Pollinations.ai — completely free, no key.
    Falls back to simpler URL if first attempt fails.
    Returns (base64_png, enhanced_prompt).
    """
    enhanced = enhance_image_prompt(prompt)
    print(f"[IMG] Prompt: {enhanced[:80]}...")

    # ── Attempt 1: Full Pollinations URL with FLUX ──
    try:
        enc     = requests.utils.quote(enhanced)
        seed    = int(time.time()) % 99999
        url     = (
            f"https://image.pollinations.ai/prompt/{enc}"
            f"?width={min(width,1024)}&height={min(height,1024)}"
            f"&model=flux&nologo=true&enhance=true&seed={seed}"
        )
        print(f"[IMG] Trying Pollinations FLUX...")
        resp = requests.get(url, timeout=90)
        if resp.status_code == 200 and "image" in resp.headers.get("content-type",""):
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img = ImageEnhance.Sharpness(img).enhance(1.25)
            img = ImageEnhance.Contrast(img).enhance(1.08)
            img = ImageEnhance.Color(img).enhance(1.1)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            buf.seek(0)
            print(f"[IMG] Success — {img.width}x{img.height}px")
            return base64.b64encode(buf.read()).decode(), enhanced
        print(f"[IMG] Attempt 1 failed: status {resp.status_code}")
    except Exception as e:
        print(f"[IMG] Attempt 1 error: {e}")

    # ── Attempt 2: Simpler URL, shorter prompt ──
    try:
        enc2 = requests.utils.quote(prompt[:150])
        url2 = f"https://image.pollinations.ai/prompt/{enc2}?width=512&height=512&nologo=true"
        print(f"[IMG] Trying Pollinations simple URL...")
        resp2 = requests.get(url2, timeout=60)
        if resp2.status_code == 200 and "image" in resp2.headers.get("content-type",""):
            img = Image.open(io.BytesIO(resp2.content)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            print("[IMG] Attempt 2 success")
            return base64.b64encode(buf.read()).decode(), prompt
        print(f"[IMG] Attempt 2 failed: status {resp2.status_code}")
    except Exception as e:
        print(f"[IMG] Attempt 2 error: {e}")

    # ── Final fallback ──
    print("[IMG] All attempts failed — returning placeholder")
    return generate_placeholder(prompt), prompt

# ================================================================
#  IMAGE UPSCALING — Lanczos + Unsharp Mask (no API needed)
# ================================================================

def upscale_image(image_b64, scale=2):
    data = base64.b64decode(image_b64)
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    nw, nh = img.width * scale, img.height * scale
    up  = img.resize((nw, nh), Image.LANCZOS)
    up  = ImageEnhance.Sharpness(up).enhance(1.5)
    up  = up.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    buf = io.BytesIO()
    up.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), nw, nh

# ================================================================
#  APP OPENER — opens any installed app on Windows
# ================================================================

KNOWN_APPS = {
    "chrome":             r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google chrome":      r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "firefox":            r"C:\Program Files\Mozilla Firefox\firefox.exe",
    "edge":               r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "microsoft edge":     r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "notepad":            "notepad.exe",
    "calculator":         "calc.exe",
    "paint":              "mspaint.exe",
    "word":               "winword.exe",
    "excel":              "excel.exe",
    "powerpoint":         "powerpnt.exe",
    "outlook":            "outlook.exe",
    "teams":              r"C:\Users\{user}\AppData\Local\Microsoft\Teams\current\Teams.exe",
    "discord":            r"C:\Users\{user}\AppData\Local\Discord\app-*\Discord.exe",
    "spotify":            r"C:\Users\{user}\AppData\Roaming\Spotify\Spotify.exe",
    "steam":              r"C:\Program Files (x86)\Steam\Steam.exe",
    "vlc":                r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    "vscode":             r"C:\Users\{user}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "visual studio code": r"C:\Users\{user}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "cmd":                "cmd.exe",
    "terminal":           "wt.exe",
    "powershell":         "powershell.exe",
    "explorer":           "explorer.exe",
    "file explorer":      "explorer.exe",
    "task manager":       "taskmgr.exe",
    "control panel":      "control.exe",
    "settings":           "ms-settings:",
    "camera":             "microsoft.windows.camera:",
    "photos":             "ms-photos:",
    "snipping tool":      "snippingtool.exe",
    "obs":                r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
    "zoom":               r"C:\Users\{user}\AppData\Roaming\Zoom\bin\Zoom.exe",
    "whatsapp":           r"C:\Users\{user}\AppData\Local\WhatsApp\WhatsApp.exe",
    "telegram":           r"C:\Users\{user}\AppData\Roaming\Telegram Desktop\Telegram.exe",
    "blender":            r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    "photoshop":          r"C:\Program Files\Adobe\Adobe Photoshop 2024\Photoshop.exe",
    "premiere":           r"C:\Program Files\Adobe\Adobe Premiere Pro 2024\Adobe Premiere Pro.exe",
    "after effects":      r"C:\Program Files\Adobe\Adobe After Effects 2024\Support Files\AfterFX.exe",
    "illustrator":        r"C:\Program Files\Adobe\Adobe Illustrator 2024\Support Files\Contents\Windows\Illustrator.exe",
    "epic games":         r"C:\Program Files (x86)\Epic Games\Launcher\Portal\Binaries\Win32\EpicGamesLauncher.exe",
    "epic":               r"C:\Program Files (x86)\Epic Games\Launcher\Portal\Binaries\Win32\EpicGamesLauncher.exe",
    "unity":              r"C:\Program Files\Unity Hub\Unity Hub.exe",
    "minecraft":          r"C:\Users\{user}\AppData\Roaming\.minecraft\launcher\minecraft-launcher.exe",
    "valorant":           r"C:\Riot Games\VALORANT\live\VALORANT.exe",
    "fortnite":           r"C:\Program Files\Epic Games\Fortnite\FortniteGame\Binaries\Win64\FortniteClient-Win64-Shipping.exe",
}

def resolve_path(path):
    user = os.environ.get("USERNAME", "User")
    path = path.replace("{user}", user)
    if "*" in path:
        matches = glob.glob(path)
        return matches[0] if matches else path
    return path

def open_app(name):
    n = name.lower().strip()
    if n in KNOWN_APPS:
        path = resolve_path(KNOWN_APPS[n])
        try:
            if any(path.startswith(p) for p in ["ms-","bing","outlook","microsoft."]):
                os.startfile(path)
            else:
                subprocess.Popen(path, shell=True)
            return f"Opening {name}."
        except Exception as e:
            print(f"App error: {e}")
    try:
        subprocess.Popen(f'start "" "{name}"', shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Launching {name}."
    except:
        pass
    for d in [r"C:\Program Files", r"C:\Program Files (x86)",
              os.path.expanduser(r"~\AppData\Local"),
              os.path.expanduser(r"~\AppData\Roaming")]:
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().startswith(n.split()[0]) and f.endswith(".exe"):
                    try:
                        subprocess.Popen(os.path.join(root, f))
                        return f"Found and opening {f}."
                    except:
                        pass
            if root.count(os.sep) - d.count(os.sep) > 3:
                break
    return f"Could not find {name}. Make sure it's installed."

# ================================================================
#  UTILITIES
# ================================================================

def get_indian_time():
    tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    return datetime.datetime.now(tz).strftime("%I:%M %p")

def internet_search(query):
    try:
        q    = re.sub(r'^search\s*', '', query.strip(), flags=re.I)
        data = requests.get(
            f"https://api.duckduckgo.com/?q={q}&format=json", timeout=5
        ).json()
        return data.get("AbstractText") or "No result found. Try rephrasing."
    except:
        return "Search failed."

def store_info(text):
    for k in ["my name is","i like","i am","remember that","remember"]:
        if k in text.lower():
            memory.append(text)
            save_memory(memory)
            return "Stored. I'll remember that."
    return None

# ================================================================
#  IMAGE INTENT DETECTION — catches all natural ways to ask
# ================================================================

IMAGE_TRIGGERS = re.compile(
    r'\b(generate|create|make|draw|render|show|give me|produce|design|paint|'
    r'sketch|illustrate|build)\b.{0,30}?\b(image|photo|picture|pic|art|'
    r'illustration|drawing|painting|portrait|wallpaper|thumbnail|poster|logo|'
    r'icon|scene|landscape|character|anime|realistic|3d|cartoon|city|nature|'
    r'space|animal|human|robot|dragon|fantasy|monster|dog|cat|bird|car|'
    r'house|sunset|mountain|ocean|forest|sky|person|face|background)\b',
    re.I | re.S
)

IMAGE_OF = re.compile(
    r'\b(image|photo|picture|pic|art|illustration|drawing|painting)\s+of\b', re.I
)

def detect_image_intent(text):
    if IMAGE_OF.search(text):
        return text
    if IMAGE_TRIGGERS.search(text):
        return text
    return None

# ================================================================
#  SYSTEM PROMPT
# ================================================================

SYSTEM_PROMPT = """You are Aeronix — a next-gen AI assistant for gamers, creators, scientists, and developers.

PERSONALITY: Direct, sharp, zero fluff. Like a pro gamer with a PhD who explains things simply.

━━ GAMING MODE ━━
Topics: tactics, meta, builds, strategy, esports, game mechanics, competitive play.
Give TIER-S level advice — specific loadouts, rotations, callouts, win conditions.
Cover FPS, MOBA, RPG, RTS, battle royale, MMO. Patch meta, psychology, team comps.

━━ VIDEO PROMPT MODE ━━
For: Veo, Veo2, Veo3, Runway, Sora, Kling, Pika, Luma, Adobe Firefly, Hailuo, Minimax.
ALWAYS use: [SHOT TYPE] · [SUBJECT+ACTION] · [ENVIRONMENT] · [LIGHTING] · [CAMERA MOVEMENT] · [MOOD] · [STYLE] · [SPECS]
Tailor to each tool — Veo3: photorealism/physics, Runway: camera control, Sora: world consistency, Kling: fluid motion.

━━ SCIENCE & MATH MODE ━━
Class doubts → analogy first, then precise definition.
High level → full derivation, equations, real application.
Structure: Concept → Why it works → Formula → Example → Insight.

━━ CODE MODE ━━
Writing code:
- Complete, working, production-quality code ALWAYS
- Wrap ALL code in proper ```language blocks
- Add comments explaining each section
- Follow best practices for the language
- Include imports, error handling, example usage

Single line analysis:
→ Language | Library | Function | What it does | Watch out for

Never re-read code you wrote. Be surgical.

━━ IMAGE MODE ━━
When user asks to generate an image — briefly confirm what you're creating.
The system handles actual generation automatically.
After generation, offer: upscaling, style variations, video prompt based on image.

RULES:
- No filler words or padding
- Code always in ``` blocks
- Math: formula + intuition
- Keep responses tight — system reads them aloud"""

# ================================================================
#  MODE DETECTION
# ================================================================

def detect_mode(text):
    t = text.lower()
    code_sigs = ['(',')','{}','[]','=>','==','!=','+=','def ','import ',
                 'from ','var ','let ','const ','return ','class ','#include','::','->']
    is_single_code = (
        '\n' not in text.strip() and
        any(s in text for s in code_sigs) and
        not any(text.lower().startswith(w) for w in
                ['what','why','how','when','open','search','create','make','generate','write','build'])
    )
    if is_single_code:
        return "code_line"
    if any(w in t for w in ["prompt for","veo","runway","sora","kling","pika","luma","firefly","hailuo","video prompt"]):
        return "video"
    if any(w in t for w in ["tactic","strategy","build","loadout","meta","ranked","competitive","esport","fps","moba","rts","boss","loot","grind","game","weapon","armor","respawn","rotation"]):
        return "gaming"
    if any(w in t for w in ["theorem","equation","integral","derivative","quantum","formula","proof","atom","molecule","force","velocity","entropy","calculus","algebra","biology","chemistry","physics","doubt","explain how","what is","how does"]):
        return "science"
    if any(w in t for w in ["write code","write a","create a script","build me","code for","python","javascript","html","css","react","flask","function","algorithm","program","script","write me a"]):
        return "code_write"
    return "general"

# ================================================================
#  GROQ STREAMING — speaks first sentence immediately
# ================================================================

def groq_stream(user_input, image_b64=None, file_text=None, mode=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    if memory:
        msgs.append({"role": "system", "content": "Recent context:\n" + "\n".join(memory[-4:])})

    mode_hints = {
        "code_line":  "Single line of code. Format: Language | Library | Function | What it does | Watch out for.",
        "code_write": "Write complete working code with ```language blocks, comments, imports, error handling.",
        "science":    "Structure: Concept → Why → Formula → Example → Insight. Use analogies.",
        "gaming":     "Tier-S tactical advice. Specific, not generic.",
        "video":      "Full 8-part structure. Tool-specific.",
    }
    if mode in mode_hints:
        msgs.append({"role": "system", "content": mode_hints[mode]})

    if image_b64:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": user_input or "Analyze deeply. Generate optimized video AI prompts. Suggest best tools."}
        ]
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
    elif file_text:
        content = f"File content:\n{file_text[:3000]}\n\nInstruction: {user_input}"
        model   = "llama-3.3-70b-versatile"
    else:
        content = user_input
        model   = "llama-3.3-70b-versatile"

    msgs.append({"role": "user", "content": content})

    stream = client.chat.completions.create(
        model=model, messages=msgs,
        max_tokens=1200, temperature=0.72, stream=True
    )

    full_reply   = []
    first_spoken = False
    sentence_buf = []

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue
        full_reply.append(token)
        yield token, False

        # Speak first sentence the moment it completes
        if not first_spoken:
            sentence_buf.append(token)
            combined = "".join(sentence_buf)
            if any(p in combined for p in [". ","? ","! ",".\n","?\n","!\n"]):
                first = combined.strip()
                if len(first) > 8:
                    speak_bg(first)
                    first_spoken = True

    reply = "".join(full_reply).strip()
    if not first_spoken and reply:
        speak_bg(reply)

    clean_reply = re.sub(r'```[\s\S]*?```', '[code]', reply)
    memory.append(f"User: {user_input[:100]}")
    memory.append(f"Aeronix: {clean_reply[:180]}")
    if len(memory) > 60:
        memory[:] = memory[-60:]
    save_memory(memory)
    yield "", True

# ================================================================
#  COMMAND HANDLER — instant, no AI
# ================================================================

def handle_command(text):
    u = text.lower().strip()
    if u in ("exit","quit","shutdown","bye"):
        return "Shutting down. GG."
    if re.match(r'^(what.s the |what is the |)time$', u):
        return f"It's {get_indian_time()}."
    if u.startswith("open "):
        return open_app(u[5:].strip())
    if re.match(r'^search\s+', u):
        return internet_search(u)
    mem = store_info(text)
    if mem:
        return mem
    return None

# ================================================================
#  FLASK ROUTES
# ================================================================

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f     = request.files["file"]
    fname = f.filename
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    f.save(fpath)
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    if ext in ("jpg","jpeg","png","gif","webp","bmp"):
        with open(fpath,"rb") as img:
            b64 = base64.b64encode(img.read()).decode()
        return jsonify({"type":"image","name":fname,"data":b64})
    try:
        with open(fpath,"r",encoding="utf-8",errors="ignore") as tf:
            text = tf.read(6000)
        return jsonify({"type":"text","name":fname,"data":text})
    except:
        return jsonify({"type":"unknown","name":fname,"data":""})


@app.route("/upscale-image", methods=["POST"])
def upscale_route():
    data  = request.get_json(silent=True) or {}
    b64   = data.get("image","")
    scale = int(data.get("scale", 2))
    if not b64:
        return jsonify({"error":"No image"}), 400
    try:
        result, w, h = upscale_image(b64, scale)
        return jsonify({"image":result,"width":w,"height":h})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/stream", methods=["POST"])
def stream_chat():
    data       = request.get_json(silent=True) or {}
    user_input = data.get("message","").strip()
    image_b64  = data.get("image")
    file_text  = data.get("file_text")

    if not user_input and not image_b64 and not file_text:
        return jsonify({"reply":"Nothing received."})

    # ── IMAGE GENERATION ──
    img_prompt = detect_image_intent(user_input) if user_input else None
    if img_prompt and not image_b64:
        def img_gen_stream():
            yield f"data: {json.dumps({'token': f'Generating: {user_input}...', 'done': False})}\n\n"
            try:
                b64, enhanced = generate_image(img_prompt)
                msg = f"Done. Prompt used: {enhanced[:80]}"
                yield f"data: {json.dumps({'token': msg, 'done': False, 'image': b64})}\n\n"
                speak_bg("Image generated.")
            except Exception as e:
                yield f"data: {json.dumps({'token': f'Generation failed: {e}', 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        return Response(stream_with_context(img_gen_stream()), mimetype="text/event-stream")

    # ── UPSCALE ──
    if re.search(r'\bupscale\b', user_input, re.I) and image_b64:
        scale = 4 if "4x" in user_input else 2
        try:
            result, w, h = upscale_image(image_b64, scale)
            def up_stream():
                msg = f"Upscaled {scale}x → {w}×{h}px using Lanczos + Unsharp Mask."
                yield f"data: {json.dumps({'token': msg, 'done': False, 'image': result})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
            speak_bg(f"Upscaled {scale}x.")
            return Response(stream_with_context(up_stream()), mimetype="text/event-stream")
        except:
            pass

    # ── FAST COMMANDS ──
    cmd = handle_command(user_input) if user_input else None
    if cmd:
        speak_bg(cmd)
        def quick():
            yield f"data: {json.dumps({'token': cmd, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        return Response(stream_with_context(quick()), mimetype="text/event-stream")

    # ── AI STREAMING ──
    mode = detect_mode(user_input) if user_input else "general"
    def ai_gen():
        try:
            for token, done in groq_stream(user_input, image_b64, file_text, mode):
                yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'token': f'Error: {e}', 'done': True})}\n\n"
    return Response(stream_with_context(ai_gen()), mimetype="text/event-stream")


# ================================================================
#  START — ngrok tunnel (FIXED) + auto browser open
# ================================================================

if __name__ == "__main__":
    public_url = None

    try:
        from pyngrok import ngrok, conf

        # Set auth token if provided
        if NGROK_TOKEN and NGROK_TOKEN != "YOUR_NGROK_TOKEN_HERE":
            conf.get_default().auth_token = NGROK_TOKEN

        # FIX: use addr= keyword so port is passed correctly
        tunnel     = ngrok.connect(addr=f"localhost:{PORT}", proto="http")
        public_url = tunnel.public_url

        # Force https
        if public_url.startswith("http://"):
            public_url = public_url.replace("http://", "https://", 1)

        print("=" * 60)
        print(f"  AERONIX v5 ONLINE")
        print(f"  Local  → http://localhost:{PORT}")
        print(f"  Public → {public_url}  ← share this!")
        print("=" * 60)

    except ImportError:
        print("=" * 60)
        print(f"  AERONIX v5 — LOCAL ONLY")
        print(f"  http://localhost:{PORT}")
        print(f"  Run: pip install pyngrok  for public URL")
        print("=" * 60)

    except Exception as e:
        print(f"  ngrok error: {e}")
        print(f"  Running locally at http://localhost:{PORT}")

    # Auto-open browser (public URL if ngrok worked, else local)
    open_url = public_url if public_url else f"http://localhost:{PORT}"

    def open_browser():
        time.sleep(1.8)   # wait for Flask to fully start
        webbrowser.open(open_url)
        print(f"  Browser → {open_url}")

    threading.Thread(target=open_browser, daemon=True).start()

    speak("Aeronix v5 online. All systems ready.")

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

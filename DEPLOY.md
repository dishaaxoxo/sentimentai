# 🚀 SentimentAI — Deploy Guide (5 minutes)

## What changed
- Removed `torch` and `transformers` (was ~1.5GB RAM)
- Now calls Hugging Face's free hosted API instead
- `requirements.txt` is now just 4 tiny packages (~10MB)
- **Render free tier will work fine**

---

## Step 1 — Get a Free HF Token (2 min)

1. Go to https://huggingface.co/join and create a free account
2. Go to https://huggingface.co/settings/tokens
3. Click **New token** → name it anything → Role: **Read** → Create
4. Copy the token (starts with `hf_...`) — save it somewhere

---

## Step 2 — Push to GitHub (1 min)

Put these 4 files in a GitHub repo:
- `app.py`
- `requirements.txt`
- `dashboard.html`
- `render.yaml` (optional but handy)

```bash
git init
git add app.py requirements.txt dashboard.html render.yaml
git commit -m "deploy: switch to HF inference API"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## Step 3 — Deploy on Render (2 min)

1. Go to https://render.com and sign in
2. Click **New** → **Web Service**
3. Connect your GitHub repo
4. Fill in:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
5. Click **Advanced** → **Add Environment Variable**
   - Key: `HF_TOKEN`
   - Value: `hf_your_token_here`
6. Click **Create Web Service**

✅ Done! Render will build and deploy in ~2 minutes.

---

## Your live URLs

Once deployed, your app will be at:
```
https://sentimentai.onrender.com/          ← Dashboard
https://sentimentai.onrender.com/docs      ← Swagger API docs
https://sentimentai.onrender.com/analyze   ← POST endpoint
https://sentimentai.onrender.com/stats     ← KPIs
```

---

## ⚠️ One thing to know

The HF model "cold starts" if it hasn't been used in a while — first request
may take ~20 seconds and return a 503. The app handles this gracefully and
tells the user to retry. Subsequent requests are instant.

---

## Alternative: Deploy on Railway

If you prefer Railway (better free tier, no sleep):
1. Go to https://railway.app
2. New Project → Deploy from GitHub repo
3. Add env var `HF_TOKEN`
4. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

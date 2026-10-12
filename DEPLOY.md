# Deploying Ashvini for free

Four free services, about forty minutes the first time.

| Part | Service | What it does |
|---|---|---|
| Database | Neon | Postgres for objects, events, decisions, alerts |
| Pipeline | GitHub Actions | Downloads orbits and space weather every four hours, screens, scores, alerts |
| API | Render | Serves the data; sleeps after 15 idle minutes on the free plan |
| Website | Vercel | The 3D console |

## 1. Put the code on GitHub

```bash
git init && git add . && git commit -m "Ashvini"
git branch -M main
git remote add origin https://github.com/Mahendr99ar/ashvini.git
git push -u origin main
```

CI runs on the first push: backend tests, ML smoke tests, web tests and build.

## 2. Database on Neon

1. Sign in at neon.tech, create a project (pick the region nearest to you).
2. Copy the connection string from **Connect**. It looks like `postgresql://user:password@host/neondb?sslmode=require`.
3. Keep it private. If it ever ends up in a chat, a screenshot or a commit, reset the password in Neon.

## 3. First pipeline run

In the GitHub repository: **Settings → Secrets and variables → Actions**.

- New secret `ASHVINI_DATABASE_URL`: the Neon string.
- New secret `ASHVINI_SECRET`: any long random text (it salts API key hashes).

Then **Actions → Refresh data → Run workflow**. The first run takes a few minutes. After that it runs every four hours by itself.

## 4. API on Render

1. render.com → **New → Blueprint** → pick the repository. Render reads `render.yaml`.
2. When asked, paste the Neon string into `ASHVINI_DATABASE_URL`, and into `ASHVINI_SECRET` the same text you gave the GitHub secret. No quotes, no spaces. The secret salts API key hashes, so if the two differ, keys made with the admin tool will not work on the website.
3. Wait for the deploy, then open `https://<your-service>.onrender.com/api/health`. You should see `"status": "ok"` and an object count in the thousands.

## 5. Website on Vercel

1. vercel.com → **Add New → Project** → import the repository.
2. **Root Directory:** `web`. Vercel detects Vite.
3. **Environment variable:** `VITE_API_URL` = your Render URL, without a trailing slash.
4. Deploy. Then, on Render, set `ASHVINI_CORS_ORIGINS` to your Vercel URL plus `http://localhost:5173`, and redeploy the API.

Refresh the snapshot the site shows while the API wakes up:

```bash
cd web && VITE_API_URL=https://<your-service>.onrender.com npm run sample
git add public/sample.json && git commit -m "Refresh sample" && git push
```

## 6. Your own organisation and keys

Run these on your machine with the Neon string, from `backend/`:

```bash
export ASHVINI_DATABASE_URL="postgresql://..."      # PowerShell: $env:ASHVINI_DATABASE_URL="postgresql://..."
export ASHVINI_SECRET="the same value as the GitHub secret"
python -m ashvini.admin create-org "Demo Ops"
python -m ashvini.admin create-key "Demo Ops" --role analyst
python -m ashvini.admin watch "Demo Ops" 25544 48274   # ISS and the Chinese station
```

Watched objects are screened against everything on the next run. The analyst key lets you record decisions on the website.

## 7. The trained models

The repository already carries the models from the Colab runs in `backend/models/`, and `render.yaml` and the refresh workflow point `ASHVINI_RISK_MODEL_PATH` at `models/collision-risk.onnx`. The API serves it only while its model card says the serving model beat the baseline; otherwise it shows the current Pc.

To retrain, open `ml/notebooks/colab_train_all.ipynb` in Google Colab (File, Open notebook, GitHub tab, paste your repo URL), switch the runtime to T4 GPU and run all. Data and results are saved in your Google Drive under `MyDrive/ashvini/`. Delete a module's `.json` card there to retrain it. Copy the new files into `backend/models/`, commit and push.

## Running it all locally instead

```bash
cp .env.example .env
docker compose up -d db api web
docker compose run --rm pipeline
# API on http://localhost:8000/docs, site on http://localhost:5173
```

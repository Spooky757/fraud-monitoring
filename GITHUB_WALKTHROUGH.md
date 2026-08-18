# How to Put Your Monitoring Repo on GitHub (Step by Step)

## What you need before starting

- A GitHub account (you said you have one)
- Git installed on your computer (confirmed — you have it)
- Your `fraud-monitoring` folder (already has commits — we're good)

---

## The Big Picture

Right now your setup looks like this:

```
YOUR COMPUTER                          GITHUB (online)
─────────────                          ──────────────
fraud-monitoring/                      (nothing yet)
  ├── src/
  ├── configs/
  ├── .github/workflows/
  └── ...
```

After this walkthrough, it will look like this:

```
YOUR COMPUTER                          GITHUB (online)
─────────────                          ──────────────
fraud-monitoring/  ◄── synced ──►  YourUsername/fraud-monitoring
  ├── src/                             ├── src/
  ├── configs/                         ├── configs/
  ├── .github/workflows/               ├── .github/workflows/  ← these run AUTOMATICALLY
  └── ...                              └── ...
```

The `.github/workflows/` folder is special — GitHub reads those YAML files and runs them automatically (daily, monthly, or when code changes). That's how monitoring and retraining happen without you doing anything.

---

## Step 1: Create an empty repo on GitHub

1. Go to **https://github.com** and log in
2. Click the **+** button (top right corner) → **New repository**
3. Fill in:
   - **Repository name**: `fraud-monitoring`
   - **Description**: `Drift detection and retraining for credit card fraud model`
   - **Visibility**: Public (or Private if you prefer)
   - **DO NOT** check "Add a README" or "Add .gitignore" (your folder already has these)
4. Click **Create repository**

You'll see a page with instructions. **Don't close this page** — you need the URL from it.

---

## Step 2: Connect your local folder to GitHub

Open a **terminal** (Command Prompt, PowerShell, or Git Bash on Windows) and run these commands one at a time:

```bash
# Navigate to your monitoring repo folder
cd "C:\Users\mypro\Dropbox\Study\Sunway Master in AI\MLOPs\Assessments\Credit Card Fraud\Retraining\fraud-monitoring\fraud-monitoring"

# Tell Git where GitHub is (replace YOUR_USERNAME with your actual GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/fraud-monitoring.git

# Push everything to GitHub
git branch -M main
git push -u origin main
```

It will ask for your GitHub username and password. **Important**: GitHub no longer accepts your regular password. You need a **Personal Access Token** instead (Step 3 below).

---

## Step 3: Create a Personal Access Token (your "password" for Git)

1. Go to **https://github.com/settings/tokens**
2. Click **Generate new token** → **Generate new token (classic)**
3. Fill in:
   - **Note**: `fraud-monitoring`
   - **Expiration**: 90 days (enough for your assessment)
   - **Scopes**: Check ☑ `repo` (this gives full access to your repos)
4. Click **Generate token**
5. **COPY THE TOKEN NOW** — you won't see it again!

When Git asks for your password in Step 2, **paste this token** instead of your password.

---

## Step 4: Verify it worked

1. Go to `https://github.com/YOUR_USERNAME/fraud-monitoring`
2. You should see all your files there
3. Click on `.github/workflows/` — you should see `ci.yml`, `monitor.yml`, and `retrain.yml`

---

## Step 5: Do the same for the training repo (if your teammate hasn't already)

The training repo (`Credit-Card-Fraud-testing`) is already on GitHub at `Koneko1625/Credit-Card-Fraud-testing`. If that's your teammate's repo, you just need to:

1. **Fork it** (click "Fork" on their repo page) — this makes your own copy
2. Add the `release.yml` workflow file I gave you into your fork's `.github/workflows/` folder

If you're using a fresh copy instead:

```bash
cd "path\to\Credit-Card-Fraud-testing"
git remote add origin https://github.com/YOUR_USERNAME/Credit-Card-Fraud-testing.git
git branch -M main
git push -u origin main
```

---

## Step 6: Add the release.yml to the training repo

This is the file that publishes model artifacts so the monitoring repo can pull them automatically.

1. Go to your training repo on GitHub
2. Click **Add file** → **Create new file**
3. In the filename box, type: `.github/workflows/release.yml`
   (GitHub automatically creates the folders when you type the slashes)
4. Paste the contents of the `training-repo-release.yml` file I gave you
5. Click **Commit changes**

---

## Step 7: Tell the monitoring repo where to find the training repo

1. Go to your `fraud-monitoring` repo on GitHub
2. Click **Settings** (tab at the top)
3. In the left sidebar: **Secrets and variables** → **Actions**
4. Click the **Variables** tab
5. Click **New repository variable**
6. Add:
   - **Name**: `TRAINING_REPO`
   - **Value**: `YOUR_USERNAME/Credit-Card-Fraud-testing`  (or `Koneko1625/Credit-Card-Fraud-testing` if using your teammate's)
7. Click **Add variable**

---

## Step 8: Test that automation works

1. Go to your `fraud-monitoring` repo on GitHub
2. Click the **Actions** tab
3. You should see three workflows listed on the left: CI, Monitor, Retrain
4. Click **Monitor** → **Run workflow** → **Run workflow**
5. Watch it run! Click on the running job to see the live logs.

---

## What happens automatically after this

| What | When | What it does |
|---|---|---|
| **CI** | Every push/PR | Runs tests to make sure nothing is broken |
| **Monitor** | Daily 2:00 AM UTC | Downloads model from training repo, scores a batch, checks for drift |
| **Retrain** | Monthly (or when monitor says RETRAIN) | Trains a new model, compares to current one, opens a PR if better |

You don't need to do anything — these run on GitHub's servers automatically. You'll see results in the **Actions** tab and get notified via GitHub Issues if something needs attention.

---

## Quick reference: common Git commands

| What you want to do | Command |
|---|---|
| See what changed | `git status` |
| Save your changes | `git add .` then `git commit -m "describe what you changed"` |
| Push to GitHub | `git push` |
| Pull teammate's changes | `git pull` |

---

## If something goes wrong

- **"Permission denied"**: Your token expired or is wrong — generate a new one (Step 3)
- **"Repository not found"**: Check the URL — typos in username/repo name
- **"Updates were rejected"**: Someone else pushed changes — run `git pull` first, then `git push`
- **Actions tab is empty**: Make sure the `.github/workflows/` folder and YAML files are on GitHub

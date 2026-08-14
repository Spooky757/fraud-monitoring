# Pushing this to a new GitHub repo

This bundle is already a git repository with one commit on `main`.

## 1. Create the repo

Either on github.com (New repository → name it `fraud-monitoring` → **do not** add a
README, .gitignore, or licence), or with the CLI:

```bash
gh repo create fraud-monitoring --private --source=. --remote=origin --push
```

## 2. Push (if you created it on the website)

```bash
cd fraud-monitoring
git remote add origin https://github.com/<your-username>/fraud-monitoring.git
git push -u origin main
```

## 3. Repo settings the workflows expect

**Settings → Actions → General → Workflow permissions**
→ "Read and write permissions" (the monitor workflow commits its state file)
→ tick "Allow GitHub Actions to create and approve pull requests"

**Settings → Secrets and variables → Actions → Variables**

| Name | Value |
|---|---|
| `TRAINING_REPO` | `Koneko1625/Credit-Card-Fraud-testing` |
| `TRAINING_REF` | `main` (pin to a tag for reproducible retrains) |

**Secrets** — only needed if the training repo is private:

| Name | Value |
|---|---|
| `TRAINING_REPO_TOKEN` | a PAT with `repo` read access |

**Labels** — create `model-monitoring`, `automated`, and `model-promotion`, or the
alert steps will fail when they try to apply them.

## 4. Verify

```bash
make demo                     # end-to-end walkthrough, no data download needed
gh workflow run monitor.yml   # or use the Actions tab → Monitor → Run workflow
```

## 5. On the training-repo side

The workflows fetch the champion from that repo's **release assets**. Attach
`model.pkl`, `scaler.pkl`, and `threshold.json` to a release there (a one-line
addition to its pipeline or a manual upload). Until you do, the workflows fall back
to whatever is committed in `artifacts/champion/` and log a warning.

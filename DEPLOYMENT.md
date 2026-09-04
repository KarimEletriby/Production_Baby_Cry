# Deployment Guide

Recommended platform for team testing: Hugging Face Spaces with Docker.

This project contains large model checkpoints, so use Git LFS when uploading the
repo. The API will be available as a public or private HTTPS URL depending on
your Space visibility.

## 1. Prepare the repo

```powershell
cd C:\BabyCry
git lfs install
git lfs track "models/*.ckpt"
git add .gitattributes Dockerfile .dockerignore DEPLOYMENT.md
git add -f models/*.ckpt
git commit -m "Prepare Docker deployment"
```

## 2. Create a Hugging Face Space

1. Open https://huggingface.co/new-space
2. Choose Docker as the SDK.
3. Choose Public for easy team testing, or Private if the model should stay internal.
4. Create the Space.

Make sure the Space README starts with this metadata:

```yaml
---
title: Baby Cry Classifier API
sdk: docker
app_port: 7860
---
```

## 3. Push this project to the Space

Replace `<hf-username>` and `<space-name>` with your values.

```powershell
git remote add space https://huggingface.co/spaces/<hf-username>/<space-name>
git push space main
```

If your branch is named `master`, use this instead:

```powershell
git push space master:main
```

## 4. Test the deployed API

Your API URL will be:

```text
https://<hf-username>-<space-name>.hf.space
```

Useful endpoints:

```text
https://<hf-username>-<space-name>.hf.space/
https://<hf-username>-<space-name>.hf.space/docs
https://<hf-username>-<space-name>.hf.space/health
```

Test prediction from PowerShell:

```powershell
curl.exe -X POST "https://<hf-username>-<space-name>.hf.space/predict" -F "audio=@C:\BabyCry\Test records\Testing record\hungry.wav"
```

## Notes

- The first build can take a while because dependencies and model files are large.
- Free CPU hardware can work for testing, but inference may be slow. Upgrade the
  Space hardware if your team needs faster responses.
- Do not commit `kaggle.json`; it is ignored from the Docker image and Git.

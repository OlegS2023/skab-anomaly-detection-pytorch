# Lab: Train on Colab GPU → Track with MLflow → Publish to Hugging Face → Serve Free via Spaces
**Important:** Please submit your work on the branch named with your index numbers i.e. `s_1xxxxx_1yyyyyy` 

**Goal:** end‑to‑end flow you can reuse in projects:
1) Fine‑tune an image classifier in **Google Colab** (GPU).
2) Track experiments and artifacts with **MLflow**.
3) Push the winning model to the **Hugging Face Hub**.
4) Serve **free inference** via a public **Hugging Face Space (CPU Basic)** and call it from your app.

---

## Architecture (at a glance)

```
 ┌──────────┐       metrics/artifacts        ┌─────────────┐
 │  Colab   │───────────────────────────────►│   MLflow    │
 │ (GPU)    │                                │ (local file │
 │ Train    │◄─────── weights / config ──────│ or server)  │
 └────┬─────┘                                └────┬────────┘
      │   push_to_hub()                           │
      │                                           │
      ▼                                           ▼
 ┌──────────────┐    pulls model at runtime  ┌───────────────┐
 │ HF Model Hub │◄───────────────────────────│ HF Space (CPU)│
 │ (your repo)  │                            │  Gradio API   │
 └──────────────┘                            └──────┬────────┘
                                                    │(free, sleeps)
                                                    ▼
                                               your app/client
```

---

## Prerequisites

- A free **Hugging Face** account and **access token** with *write* scope: https://huggingface.co/settings/tokens  
- A **Google Colab** account. In Colab: `Runtime → Change runtime type → T4 GPU` (or any available GPU).
- (Optional) **ngrok** token if you want to view the MLflow UI from the browser.

## Validation checklist

- [ ] Colab shows **GPU** in `torch.cuda.is_available()` and training completes.
- [ ] MLflow folder `/content/mlruns` contains your run, and artifacts (CSV/PNG) exist under `eval/`.
- [ ] The model appears at `https://huggingface.co/<your-username>/<repo>` with a model card.
- [ ] Your Space builds on **CPU Basic (Free)** and the **Predict** tab returns reasonable labels.
- [ ] The **Metrics** tab loads your per-class table and shows the confusion matrix.

---

## Troubleshooting

- **401 / permission denied when pushing to Hub** — ensure you’re logged in and `hub_repo_id` uses your HF username.  
- **Model too slow on CPU Space** — switch to a smaller backbone (e.g., `vit-tiny` or a small CNN), or export to ONNX and run with `onnxruntime` in the Space.  
- **Dataset errors** — for your own data, match `imagefolder` layout: `train/`, `validation/`, `test/` with class-subfolders.  
- **Space can’t find artifacts** — verify you uploaded `assets/per_class_metrics.csv` and `assets/confusion_matrix.png` to the Space repo.  
- **Colab times out** — reduce `max_train_samples` and epochs, or persist your outputs to Drive.

---

[//]: # (## Extensions &#40;optional&#41;)

[//]: # ()
[//]: # (- Add a **FastAPI** Space that returns pure JSON &#40;no UI&#41; for cleaner app integration.  )

[//]: # (- Use **MLflow Model Registry** &#40;with a tracking server&#41; to mark “Staging/Prod” and auto-push the prod model to the Space.  )

[//]: # (- Add **explanations** &#40;e.g., Grad‑CAM images&#41; to artifacts and display them in a new Space tab.)

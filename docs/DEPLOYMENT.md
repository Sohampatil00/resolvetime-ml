# Deploy ResolveTime on Streamlit Community Cloud

The project is prepared for cloud inference with its selected trained model. These instructions are for the owner to run; no terminal command, push, or live deployment was performed while preparing this guide.

## Files the cloud app needs

- `app.py` and the `resolvetime/` package.
- `models/resolution_model.joblib`, the selected locally trained artifact (about 16 MB).
- `requirements.txt` and `requirements-lock.txt`.

`.gitignore` permits that one model file while excluding other model outputs, raw data, environments, and generated reports. The app does not train in the cloud or require the raw dataset for inference. Evaluation is embedded inside the saved artifact.

## 1. Push the prepared changes

If you already initialized the repository and configured its `origin`, run:

```bash
cd "/Users/sohampatil/Documents/ChatGPT/ML minipj/resolve_time"
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare Streamlit cloud deployment with trained model"
git push -u origin main
```

Check that the selected model is included in the staged/committed files. It was previously excluded before the deployment configuration changed. If Git initialization or the remote is missing, use the initial setup in [PUBLISHING.md](PUBLISHING.md) first. Do not force-push to resolve unexpected remote commits.

## 2. Create the app

1. Open [share.streamlit.io](https://share.streamlit.io/).
2. Sign in and connect your GitHub account if needed.
3. Click **Create app**, then choose **Yup, I have an app** if asked.
4. Enter repository **Sohampatil00/resolvetime-ml**.
5. Enter branch **main** and main file **app.py**.
6. Open **Advanced settings** and select Python **3.11**. Do not assume `.python-version` configures the cloud selector automatically.
7. No app secrets are required. Save the settings and click **Deploy**.

The initial GitHub authorization, any account agreement, and the final platform interaction can be completed by the account owner. Streamlit will install the recorded dependencies from the repository. Its deployment screen shows logs and, after successful startup, the shareable app URL.

An optional custom subdomain must be available; no particular `streamlit.app` URL is guaranteed in advance.

## 3. Verify the deployed app

1. Confirm the compact ticket form appears.
2. Select a category/subcategory, impact, and urgency, then click **Predict resolution time**.
3. Confirm a numeric hour estimate appears.
4. Change an input and verify the previous estimate clears.
5. Open **Model results** and confirm the metrics match the artifact: the reference run selected k-NN, with test MAE 147.69 hours and median-baseline MAE 144.95 hours.

Hosting does not improve model accuracy. Keep the prototype limitations and baseline comparison visible.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| App says to train first | `models/resolution_model.joblib` must exist in the GitHub branch being deployed. Local-only files are not copied to the cloud. |
| scikit-learn version mismatch | Confirm both requirements files are pushed. `requirements.txt` includes the recorded lock, including scikit-learn 1.9.1. |
| Installation failure | Read the cloud build log and confirm Python 3.11. Do not remove dependency pins randomly; the serialized model needs a compatible environment. |
| Cannot find `app.py` | Verify that the contents of `resolve_time` are the repository root, then use `app.py` as the main file path. |
| Repository is not listed | Enter owner/repository manually and check the GitHub authorization scope in Streamlit. |
| App runs locally but not in the cloud | Inspect the Streamlit deployment logs. Source code, trained model, and dependency files must all be committed and pushed. |

## References

- [Official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
- [Python dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
- [Repository file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

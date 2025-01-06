---
# see https://huggingface.co/docs/hub/spaces-config-reference

title: Online Pipeline Card Renderer
emoji: 🌖
colorFrom: yellow
colorTo: yellow
sdk: docker
suggested_hardware : "cpu-basic"
fullWidth: true
header: mini
pinned: false
license: apache-2.0
short_description: Just an html renderer for repo-hosted portable pipeline-card
tags:
  - retrain-pipelines
thumbnail: https://cdn-avatars.huggingface.co/v1/production/uploads/651e93137b2a2e027f9e55df/96hzBved0YMjCq--s0kad.png
---

<img src="https://github.com/user-attachments/assets/35cd2424-7794-411e-8367-bb82e3b96624" />


This Space serves as an API endpoint to render repo-hosted `pipeline-card` items. It complements model-versions trained with the <code><a target="_blank" href="https://github.com/aurelienmorgan/retrain-pipelines">retrain-pipelines</a></code> library, which publishes them on the Hub!

In essence, it serves the same purpose as <a target="_blank" href="https://html-preview.github.io/">html-preview.github.io</a> does for GitHub-hosted html files, but for HF-Hub hosted ones&nbsp;!

So, basically, it handles authentication and renders raw repo html files into your web-browser.

<Tip>

If you want to use it for model repositories that are <a target="_blank" href="https://huggingface.co/docs/hub/models-gated">gated</a> or private to you and published via the `retrain-pipelines` librairy, just duplicate it and assigne the `HF_TOKEN`secret with a proper value. Simple as that&nbsp;!

</Tip>


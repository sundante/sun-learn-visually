# Attention to Agents

An interactive guide to how LLMs evolved from 2017 to 2026, built from `PROMPT-v2.md`.

| Path | What it is |
|---|---|
| `attention-to-agents.html` | The interactive page. Open it in any browser; it is self-contained, with the code and video embedded. |
| `code/00–06_*.py` | Seven PyTorch 2.x modules. Each one checks itself with `assert torch.allclose` against PyTorch's reference. |
| `animation/nlp_evolution_scenes.py` | Manim CE script with three scenes: the timeline, RNN vs attention, and MoE routing. |
| `animation/nlp_evolution.mp4` | The three scenes rendered at 720p30 (about 40 seconds). |
| `diagrams.md` | Mermaid source for the four flow diagrams. |

## Run the code
```bash
pip install torch
cd code && for f in 0*.py; do python "$f"; done
```

## Render the animation
```bash
brew install cairo pango pkg-config   # macOS prerequisites
pip install manim
cd animation
manim -pqh nlp_evolution_scenes.py RNNvsAttentionScene   # 1080p60
manim -qm nlp_evolution_scenes.py -a                      # all scenes, 720p
```

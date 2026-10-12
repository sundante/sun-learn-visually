"""
NLP → LLM evolution, animated with Manim Community Edition (v0.18+).

Scenes
  1. TimelineScene         2017 → 2026 milestones, with a pre-2017 prelude stub
  2. RNNvsAttentionScene   serial RNN steps vs all-at-once self-attention
  3. MoERoutingScene       tokens routed to their top-2 of 8 experts

Install
  pip install manim          # needs Cairo + Pango (macOS: brew install cairo pango pkg-config)

Render (from this folder)
  manim -pqm nlp_evolution_scenes.py TimelineScene          # -p preview, -qm 720p30
  manim -pqh nlp_evolution_scenes.py RNNvsAttentionScene    # -qh 1080p60
  manim -qm  nlp_evolution_scenes.py MoERoutingScene
  manim -qm  nlp_evolution_scenes.py -a                     # all scenes

Only Text objects are used (no LaTeX), so no TeX install is required.
"""
from manim import *

# Light palette to match the interactive page
BG = "#F7F8FB"
INK = "#14172B"
MUTED = "#6B7090"
ACCENT = "#3B5BFD"   # indigo-blue
WARM = "#FF7A45"     # coral, for "serial / old"
MINT = "#14B88A"     # green, for "parallel / new"
FONT = "Helvetica"

config.background_color = BG


def label(text, size=24, color=INK, weight=NORMAL):
    return Text(text, font=FONT, font_size=size, color=color, weight=weight)


class TimelineScene(Scene):
    MILESTONES = [
        ("2017", "Transformer", "self-attention, fully parallel"),
        ("2018", "BERT · GPT-1", "pre-train, then fine-tune"),
        ("2020", "GPT-3 · Scaling laws", "few-shot in-context learning"),
        ("2022", "InstructGPT · Chinchilla", "RLHF, compute-optimal data"),
        ("2023", "Llama · DPO · Mixtral", "open weights, GQA, MoE"),
        ("2024", "Reasoning · MCP", "test-time compute, tool protocols"),
        ("2025", "Agentic systems", "RL on verifiable rewards, agents"),
        ("2026", "Today", "long-context, multi-agent stacks"),
    ]

    def construct(self):
        title = label("From Attention to Agents", 40, INK, BOLD).to_edge(UP, buff=0.5)
        sub = label("2017 → 2026: where compute, data and code changed", 22, MUTED).next_to(title, DOWN, buff=0.15)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))

        line = Line(LEFT * 6, RIGHT * 6, color=MUTED, stroke_width=3)
        # Pre-2017 prelude stub: a dashed segment on the left
        prelude = DashedLine(LEFT * 6.9, LEFT * 6, color=MUTED, stroke_width=3)
        prelude_txt = label("← pre-2017 prelude: BoW · word2vec · LSTM · Seq2Seq", 15, MUTED)
        prelude_txt.move_to([0, -2.4, 0]).to_edge(LEFT, buff=0.3)
        self.play(Create(prelude), FadeIn(prelude_txt), Create(line), run_time=1.2)

        xs = [-5.3 + i * (10.6 / (len(self.MILESTONES) - 1)) for i in range(len(self.MILESTONES))]
        for i, ((year, name, note), x) in enumerate(zip(self.MILESTONES, xs)):
            dot = Dot([x, 0, 0], radius=0.11, color=ACCENT)
            yr = label(year, 20, ACCENT, BOLD).next_to(dot, DOWN if i % 2 == 0 else UP, buff=0.2)
            card = VGroup(label(name, 17, INK, BOLD), label(note, 13, MUTED)).arrange(DOWN, buff=0.08)
            card.next_to(dot, UP if i % 2 == 0 else DOWN, buff=0.45)
            stem = Line(dot.get_center(), card.get_edge_center(DOWN if i % 2 == 0 else UP), color=ACCENT, stroke_width=1.5, stroke_opacity=0.5)
            self.play(GrowFromCenter(dot), FadeIn(yr), Create(stem), FadeIn(card, shift=(UP if i % 2 == 0 else DOWN) * 0.2), run_time=0.6)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


class RNNvsAttentionScene(Scene):
    WORDS = ["The", "cat", "sat", "on", "the", "mat"]

    def token_row(self, y, color):
        boxes = VGroup()
        for w in self.WORDS:
            box = RoundedRectangle(width=1.35, height=0.62, corner_radius=0.12, stroke_color=color, stroke_width=2, fill_color=WHITE, fill_opacity=1)
            boxes.add(VGroup(box, label(w, 20, INK)))
        boxes.arrange(RIGHT, buff=0.35).move_to([0, y, 0])
        return boxes

    def construct(self):
        # ---- RNN: one token at a time ----
        h1 = label("RNN / LSTM: one token per step", 28, WARM, BOLD).to_edge(UP, buff=0.4)
        row = self.token_row(1.2, WARM)
        self.play(FadeIn(h1), LaggedStart(*[FadeIn(t) for t in row], lag_ratio=0.1))
        state = Circle(radius=0.22, color=WARM, fill_opacity=0.9).next_to(row[0], DOWN, buff=0.6)
        state_lbl = label("h", 18, WHITE, BOLD).move_to(state)
        hidden = VGroup(state, state_lbl)
        step_counter = label("step 1 / 6", 20, MUTED).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(hidden), FadeIn(step_counter))
        for i, tok in enumerate(row):
            new_counter = label(f"step {i + 1} / 6", 20, MUTED).to_edge(DOWN, buff=0.6)
            self.play(
                tok[0].animate.set_fill(WARM, opacity=0.85),
                hidden.animate.next_to(tok, DOWN, buff=0.6),
                run_time=0.55,
            )
            step_counter.become(new_counter)
        note = label("h_t depends on h_(t-1): the GPU waits on the loop", 20, INK).next_to(row, DOWN, buff=1.6)
        self.play(FadeIn(note))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # ---- Transformer: every token at once ----
        h2 = label("Self-attention: every token, one matrix multiply", 28, MINT, BOLD).to_edge(UP, buff=0.4)
        row2 = self.token_row(-1.6, MINT)
        self.play(FadeIn(h2), LaggedStart(*[FadeIn(t) for t in row2], lag_ratio=0.05))
        self.play(*[t[0].animate.set_fill(MINT, opacity=0.85) for t in row2], run_time=0.6)

        arcs = VGroup()
        for i, a in enumerate(row2):
            for j, b in enumerate(row2):
                if i < j:
                    arc = ArcBetweenPoints(a.get_top(), b.get_top(), angle=-PI * 0.55,
                                           color=ACCENT, stroke_width=1.2 + 2.2 / (j - i), stroke_opacity=0.55)
                    arcs.add(arc)
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.03), run_time=1.6)
        cap = label("6 × 6 attention matrix, computed in parallel: 1 step", 20, INK).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(cap))
        self.play(arcs.animate.set_stroke(opacity=0.9), rate_func=there_and_back, run_time=1.2)
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in self.mobjects])


class MoERoutingScene(Scene):
    def construct(self):
        title = label("Mixture of Experts: each token uses 2 of 8 experts", 28, INK, BOLD).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))

        router = RoundedRectangle(width=2.2, height=0.9, corner_radius=0.15, color=ACCENT, fill_color=ACCENT, fill_opacity=0.12)
        router_lbl = label("Router", 22, ACCENT, BOLD).move_to(router)
        R = VGroup(router, router_lbl).move_to([-2.6, 0, 0])

        experts = VGroup()
        for e in range(8):
            box = RoundedRectangle(width=1.6, height=0.55, corner_radius=0.1, stroke_color=MUTED, stroke_width=1.5, fill_color=WHITE, fill_opacity=1)
            experts.add(VGroup(box, label(f"Expert {e + 1}", 16, INK)))
        experts.arrange(DOWN, buff=0.14).move_to([2.4, -0.35, 0])
        self.play(FadeIn(R), LaggedStart(*[FadeIn(x) for x in experts], lag_ratio=0.05))

        tokens = ["cat", "2+2", "def", "Paris"]
        routes = [(0, 3), (5, 6), (1, 5), (3, 7)]
        colors = [WARM, MINT, ACCENT, "#B04CE0"]
        for tok, (a, b), col in zip(tokens, routes, colors):
            t = VGroup(RoundedRectangle(width=1.1, height=0.5, corner_radius=0.1, color=col, fill_color=col, fill_opacity=0.9),
                       label(tok, 18, WHITE, BOLD))
            t[1].move_to(t[0])
            t.move_to([-5.6, 0, 0])
            self.play(FadeIn(t, shift=RIGHT * 0.3), run_time=0.35)
            self.play(t.animate.move_to(R.get_left() + LEFT * 0.7), run_time=0.4)
            l1 = Line(R.get_right(), experts[a].get_left(), color=col, stroke_width=4)
            l2 = Line(R.get_right(), experts[b].get_left(), color=col, stroke_width=4)
            self.play(Create(l1), Create(l2),
                      experts[a][0].animate.set_fill(col, opacity=0.35),
                      experts[b][0].animate.set_fill(col, opacity=0.35), run_time=0.6)
            self.wait(0.3)
            self.play(FadeOut(t), FadeOut(l1), FadeOut(l2),
                      experts[a][0].animate.set_fill(WHITE, opacity=1),
                      experts[b][0].animate.set_fill(WHITE, opacity=1), run_time=0.35)

        cap = label("Mixtral 8x7B: ~47B parameters stored, ~13B active per token", 20, MUTED).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(cap))
        self.wait(1.5)

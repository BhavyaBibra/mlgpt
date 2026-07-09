# MLGPT — UI Design

Design goal: look like a serious observability product (Datadog/Grafana energy) with one twist — a conversational agent panel that makes it feel alive. The UI is 50% of the GitHub first impression; the demo GIF is shot entirely inside it.

## Visual identity
- **Theme:** dark mode default (observability tools live in dark mode; also demos/GIFs look dramatically better)
- **Palette:** near-black background (#0B0E14), one accent (electric teal or violet), semantic green/amber/red for health states only
- **Type:** Inter or Geist; monospace (JetBrains Mono) for feature names, versions, timestamps
- **Feel:** dense but calm — generous spacing, thin 1px borders, no card shadows, subtle glow only on live-updating elements

## Layout (single-page app, left nav)

```
┌──────┬─────────────────────────────────────────────┬──────────────┐
│      │  Overview / Model detail / Incidents        │              │
│ nav  │  (main content area)                        │  Ask MLGPT   │
│      │                                             │  (chat panel,│
│      │                                             │  collapsible)│
└──────┴─────────────────────────────────────────────┴──────────────┘
```

## Screens

### 1. Overview (landing)
- **Model health cards**: one card per model — name, big status dot (🟢🟡🔴), sparkline of accuracy (7d), current drift score, "last incident 2d ago"
- Cards pulse subtly when live data arrives (WebSocket) — the "it's alive" moment
- Global stats strip: predictions today, active incidents, models watched

### 2. Model detail (the hero screen — the GIF is shot here)
- **Accuracy timeline** (main chart): line chart with **deploy markers as vertical annotated lines** ("v2.3") and **incident regions shaded red** — the moment someone sees drift start exactly at a deploy marker is the whole product in one image
- **Feature drift heatmap**: rows = features, columns = time buckets, cells colored by drift score. Instantly scannable "which feature went bad and when"
- **Prediction distribution**: two overlaid histograms (reference vs. live) with a time scrubber — drag it and watch the distribution slide apart
- **Live feed**: right side, streaming ticker of incoming predictions (anonymized) — pure demo candy but signals "real-time"

### 3. Incident view (the credibility screen)
- **Incident timeline**: vertical timeline — `14:02 deploy v2.3` → `14:40 drift detected in transaction_amount` → `14:41 agent explanation` → `resolved`
- **Agent explanation card**: the structured root-cause report rendered beautifully:
  - Summary sentence in large type
  - Evidence list where **every citation is a clickable chip** — click "PSI 0.31 transaction_amount" and the heatmap scrolls/highlights that exact cell; click "Deployed v2.3" and the timeline chart flashes that marker. **This interaction is the wow feature. Prioritize it.**
  - Confidence badge + suggested action
- Thumbs up/down on explanations (stored — future training data, also looks thoughtful)

### 4. Ask MLGPT (persistent right panel)
- Chat interface, model-scoped ("Asking about: fraud-v2")
- Answers stream in token-by-token with the same clickable citation chips
- Suggested questions as chips when empty: "Why did accuracy drop this week?" / "What changed after the last deploy?" / "Which feature is drifting most?"

### 5. Events & settings (utility screens, keep minimal)
- Events log table (filter by kind), threshold configuration per model, Slack webhook setup

## Attractive details worth the effort (ranked by ROI)
1. **Clickable evidence chips** linking explanation → chart highlight (the differentiator, film this)
2. **Deploy markers on the accuracy chart** (tells the causal story visually)
3. **Drift heatmap** (dense, professional, screenshots beautifully)
4. **Live pulse on health cards** via WebSocket
5. **Time scrubber on distribution overlay** (interactive drift intuition)
6. Streaming chat answers
7. Command palette (Cmd+K to jump between models) — cheap to add with a library, feels premium

## Anti-goals
- No 3D, no particle effects, no gradient-splash landing page — observability users trust restraint
- No modal-heavy flows; everything inline
- No login wall in the demo

## Stack
- React 18 + TypeScript + Vite
- Tailwind CSS
- Recharts (timeline, sparklines) + a heatmap (visx or custom grid — a CSS grid heatmap is simpler and fully controllable)
- WebSocket for live updates
- Framer Motion for the few animations that matter (card pulse, chip highlight)

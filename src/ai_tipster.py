"""
AI Tipster — an LLM reviews the model's own prediction, does its own
reasoning (with live web search for anything the statistical model can't
see — injuries, suspensions, manager changes, pundit takes), and produces
an independent "AI Prediction" for both the scoreline and the safe-bet
call. Shown as its own section on the match card / modal, alongside (not
replacing) the model's prediction.

Cost/architecture note — read before changing the trigger site: this is a
genuine, metered LLM call (unlike every other integration in this app,
which are all free APIs). It must run ONCE per fixture, fire-and-forget,
right after the prediction is first saved — never live on a page load,
and never re-run for a fixture that already has a row (see
database.get_ai_tip/save_ai_tip and the uniqueness on ai_tips
(fixture_id, match_date)). Triggered from database.save_predictions() /
save_basketball_predictions() alongside the existing prediction save.

Inert by default: every entry point checks config.ANTHROPIC_API_KEY first
and no-ops (status="skipped_no_key") if it isn't set — same pattern as
every other optional external dependency in this app (Supabase, TheSportsDB,
football-data.org). No key, no calls, no cost.
"""

import json
import re

from src.config import ANTHROPIC_API_KEY

# claude-opus-5 per this project's default model policy — this is a
# reasoning + synthesis task (compare the model's numbers against fresh
# web context and form a judgment), not a cheap classification, so it
# stays on the default rather than being downgraded for cost. If the
# "all leagues + basketball from day one" volume this shipped with turns
# out to be too expensive in practice, switching to "claude-sonnet-5" here
# is the one-line lever — that's a cost/quality tradeoff for whoever's
# paying the bill to make deliberately, not something to default to quietly.
MODEL = "claude-opus-5"

# Basic web_search variant (broadest model/platform compatibility) — this
# runs as a background batch job, not a latency-sensitive user-facing
# request, so there's no reason to require the newer dynamic-filtering
# variant's model restriction.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 4}

SYSTEM_PROMPT = """You are an experienced football/basketball tipster reviewing a statistical \
model's prediction for one match. You are given the model's own numbers (predicted score, win/\
draw/loss probabilities, recent form, head-to-head, team strength ratings) plus real match \
context. Use web search to check for anything the statistical model cannot see: confirmed \
injuries/suspensions, lineup news, manager changes, and current expert/pundit predictions for \
this exact fixture.

Weigh the model's numbers AND what you find, then give your own independent verdict — you may \
agree with the model or disagree with it; say which and why. Be concise: a tipster's note, not \
an essay.

End your reply with EXACTLY ONE fenced JSON block, on its own, in this exact shape (no other \
JSON blocks anywhere in the reply):

```json
{
  "ai_predicted_home": <int>,
  "ai_predicted_away": <int>,
  "ai_safe_bet_line": "<e.g. \\"2.5\\" or \\"under_0.5\\", your own pick, not necessarily the model's>",
  "ai_safe_bet_pick": "<\\"over\\" or \\"under\\">",
  "agrees_with_model": <true|false — does your predicted outcome (home/draw/away) match the model's?>,
  "confidence": <int 0-100, your own confidence in your verdict>,
  "reasoning": "<2-4 sentences, plain language, suitable to show a user directly>"
}
```"""


def _build_user_prompt(pred: dict) -> str:
    """Summarize the model's own output + underlying signal into a compact
    prompt — everything the LLM needs to form an opinion without having to
    re-derive it from raw ESPN data itself."""
    p = pred.get("prediction") or {}
    f = pred.get("features") or {}
    sport = pred.get("sport", "football")
    home, away = pred.get("home_team", ""), pred.get("away_team", "")

    lines = [
        f"Sport: {sport}",
        f"League/competition: {pred.get('league', '')}",
        f"Fixture: {home} (home) vs {away} (away)",
        f"Kickoff: {pred.get('match_time', 'unknown')}",
        f"Venue: {pred.get('venue', 'unknown')}",
        "",
        "-- The statistical model's own prediction --",
        f"Predicted score: {home} {p.get('predicted_home')} - {p.get('predicted_away')} {away}",
        f"Win/Draw/Loss probability: {p.get('win_probability')}% / "
        f"{p.get('draw_probability')}% / {p.get('loss_probability')}%",
        f"Model confidence in that exact scoreline: {p.get('confidence')}%",
    ]

    sb = p.get("safe_bet") or {}
    if sb:
        lines.append(f"Model's safe bet: {sb.get('type', 'over')} {sb.get('line')} "
                      f"({sb.get('probability')}% probability)")

    if sport == "football":
        home_r, away_r = pred.get("home_ratings") or {}, pred.get("away_ratings") or {}
        if home_r:
            lines.append(f"{home} strength rating: {home_r.get('stars')}/5 stars "
                         f"(attack {home_r.get('attack')}, midfield {home_r.get('midfield')}, "
                         f"defence {home_r.get('defence')}, all /100)")
        if away_r:
            lines.append(f"{away} strength rating: {away_r.get('stars')}/5 stars "
                         f"(attack {away_r.get('attack')}, midfield {away_r.get('midfield')}, "
                         f"defence {away_r.get('defence')}, all /100)")
        if f.get("h2h_matches"):
            lines.append(f"Head-to-head: {f['h2h_matches']} recent meetings, "
                         f"avg {f.get('h2h_home_avg')}-{f.get('h2h_away_avg')} goals")

    def _form_str(matches):
        if not matches:
            return "no recent form data"
        return ", ".join(
            f"{m.get('goals_for', m.get('pts_for'))}-{m.get('goals_ag', m.get('pts_ag'))}"
            for m in matches[:5]
        )

    lines.append(f"{home} last 5: {_form_str(pred.get('home_form'))}")
    lines.append(f"{away} last 5: {_form_str(pred.get('away_form'))}")
    lines.append("")
    lines.append(
        f"Search for current news on this specific {home} vs {away} fixture — injuries, "
        f"suspensions, lineup doubts, manager comments, expert predictions — then give your "
        f"own verdict per the format in your instructions."
    )
    return "\n".join(lines)


def _extract_json_block(text: str) -> dict:
    """Pull the single fenced ```json ... ``` block the prompt asks for."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        raise ValueError("no fenced JSON block found in AI tipster response")
    return json.loads(match.group(1))


async def generate_ai_tip(pred: dict) -> dict:
    """
    Generate one AI tip for one already-computed model prediction.

    Returns a dict matching the ai_tips schema (see supabase_schema.sql),
    always including "status". Never raises — a failure produces
    status="failed" so the caller can persist that and move on; the
    fixture just shows no AI Prediction section rather than breaking
    anything else.
    """
    if not ANTHROPIC_API_KEY:
        return {"status": "skipped_no_key"}

    try:
        import anthropic
    except ImportError:
        return {"status": "skipped_no_key"}  # package not installed

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    messages = [{"role": "user", "content": _build_user_prompt(pred)}]
    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=[WEB_SEARCH_TOOL],
            messages=messages,
        )
        # Web search is server-executed — Claude can invoke it several times
        # within one response. A very long research turn can still come back
        # as pause_turn; resend once to let it finish (see Anthropic docs'
        # own pattern for this) rather than treating it as a failure.
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            response = await client.messages.create(
                model=MODEL, max_tokens=2000, system=SYSTEM_PROMPT,
                tools=[WEB_SEARCH_TOOL], messages=messages,
            )

        text = "".join(b.text for b in response.content if b.type == "text")
        sources = [
            {"url": r.url, "title": r.title}
            for block in response.content if block.type == "web_search_tool_result"
            for r in (block.content if isinstance(block.content, list) else [])
            if hasattr(r, "url")
        ]
        parsed = _extract_json_block(text)

        return {
            "status": "done",
            "ai_predicted_home": parsed.get("ai_predicted_home"),
            "ai_predicted_away": parsed.get("ai_predicted_away"),
            "ai_safe_bet_line": str(parsed.get("ai_safe_bet_line", "")),
            "ai_safe_bet_pick": parsed.get("ai_safe_bet_pick"),
            "agrees_with_model": parsed.get("agrees_with_model"),
            "confidence": parsed.get("confidence"),
            "reasoning": parsed.get("reasoning", ""),
            "sources": sources,
        }
    except Exception as e:
        print(f"[AI Tipster error] {pred.get('home_team')} vs {pred.get('away_team')}: {e}")
        return {"status": "failed"}

"""
next-read v10 web app: blurb-first.
"""

import os
from urllib.parse import quote_plus
import requests
import gradio as gr
from dotenv import load_dotenv
from next_read import recommend_from_name

load_dotenv()

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")


SIGNAL_LABELS = {
    # Blurb signals
    "blurb": "Back-cover blurb",
    "foreword": "Wrote foreword",
    "introduction": "Wrote introduction",
    "jacket_quote": "Jacket quote",
    "praise_page": "Praise page",
    # Casual signals
    "tweet": "Tweet",
    "blog_post": "Blog post",
    "substack": "Substack",
    "instagram": "Instagram",
    "podcast_moment": "Podcast moment",
    "interview_moment": "Interview moment",
    "social_post": "Social post",
    "unknown": "Recommendation",
}


def _signal_label(sig):
    return SIGNAL_LABELS.get(sig, sig.replace("_", " ").title())


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().split()).title()


def _is_junk_author(author: str) -> bool:
    if not author:
        return True
    a = author.lower()
    return "unknown" in a or "(" in a or "?" in a


def check_password(pw):
    if pw == APP_PASSWORD and APP_PASSWORD:
        return (gr.update(visible=False), gr.update(visible=True), "")
    return (gr.update(visible=True), gr.update(visible=False), "Wrong password. Try again.")


def run_search(name, include_casual):
    name = _normalize_name(name)
    if not name:
        return "Please enter a name.", ""
    logs = []
    def logger(msg):
        logs.append(str(msg))
    try:
        result = recommend_from_name(name, include_casual=bool(include_casual), log=logger)
        return _format_output(result), "\n".join(logs)
    except Exception as e:
        return f"Error: {e}", "\n".join(logs)


def _get_isbn(title, author):
    try:
        if _is_junk_author(author):
            q = f'intitle:"{title}"'
        else:
            q = f'intitle:"{title}" inauthor:"{author}"'
        r = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={"q": q, "maxResults": 1},
            timeout=5,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return None
        ids = items[0].get("volumeInfo", {}).get("industryIdentifiers", [])
        for ident in ids:
            if ident.get("type") == "ISBN_10":
                return ident.get("identifier")
        for ident in ids:
            if ident.get("type") == "ISBN_13":
                return ident.get("identifier")
        return None
    except Exception:
        return None


def _amazon_link(title, author):
    isbn = _get_isbn(title, author)
    if isbn:
        return f"https://www.amazon.com/dp/{isbn}"
    if _is_junk_author(author):
        query = quote_plus(title)
    else:
        query = quote_plus(f"{title} {author}")
    return f"https://www.amazon.com/s?k={query}&i=stripbooks"


def _format_section(recs, section_kind):
    """section_kind is 'blurb' or 'casual'."""
    lines = []
    for i, rec in enumerate(recs, 1):
        year = rec.get("year") or "n/a"
        author = rec.get("author", "")
        display_author = author if not _is_junk_author(author) else "—"
        amazon_url = _amazon_link(rec["title"], author)
        signal_types = rec.get("signal_types", [])

        lines.append(f"### {i}. *{rec['title']}* ({year})")
        lines.append(
            f"by {display_author}  \u00b7  "
            f"<a href='{amazon_url}' target='_blank' rel='noopener'>Buy on Amazon</a>"
        )
        if signal_types:
            chips = " · ".join(f"**{_signal_label(s)}**" for s in signal_types)
            lines.append(f"Found via: {chips}")

        if rec.get("one_line"):
            lines.append(f"_{rec['one_line']}_")

        # Best quote + URL
        best_quote = ""
        best_url = ""
        for ev in rec.get("evidence", []):
            q = ev.get("quote", "")
            url = ev.get("url", "")
            if url and not (url.startswith("http://") or url.startswith("https://")):
                url = ""
            if len(q) > len(best_quote):
                best_quote = q
                best_url = url
            elif not best_url and url:
                best_url = url
        if best_quote:
            quote = best_quote
            if len(quote) > 250:
                quote = quote[:250] + "..."
            lines.append(f"> \"{quote}\"")
        if best_url:
            lines.append(f"<sub><a href='{best_url}' target='_blank' rel='noopener'>source</a></sub>")
        lines.append("")
    return "\n".join(lines)


def _format_output(result):
    lines = []
    resolved = result.get("resolved_to") or result.get("input_name", "")
    alts = result.get("alternatives", [])
    blurbs = result.get("blurbs", [])
    casual = result.get("casual_shares", [])
    include_casual = result.get("include_casual", False)

    lines.append(f"# Blurbs by {resolved}")
    if alts:
        lines.append(f"_Other people with this name: {', '.join(alts)}_")
    lines.append("")

    if not blurbs:
        lines.append(f"_{resolved} has not written blurbs for any books, yet._")
        lines.append("")
    else:
        lines.append(f"_Found {len(blurbs)} books with formal endorsements._")
        lines.append("")
        lines.append(_format_section(blurbs, "blurb"))

    if include_casual:
        lines.append("---")
        lines.append("")
        lines.append(f"# Casual shares by {resolved}")
        lines.append(
            "_Tweets, blog posts, Substack, podcasts, and other places "
            "{name} has personally recommended a book._".replace("{name}", resolved)
        )
        lines.append("")
        if not casual:
            lines.append(f"_No casual shares found for {resolved}._")
        else:
            lines.append(f"_Found {len(casual)} casual recommendations._")
            lines.append("")
            lines.append(_format_section(casual, "casual"))

    return "\n".join(lines)


with gr.Blocks(title="NextRead") as demo:
    with gr.Column(visible=True) as login_section:
        gr.Markdown("# NextRead")
        gr.Markdown("Please enter the password to access this app.")
        pw_input = gr.Textbox(label="Password", type="password")
        pw_btn = gr.Button("Enter", variant="primary")
        pw_error = gr.Markdown("")

    with gr.Column(visible=False) as app_section:
        gr.Markdown("# NextRead")
        gr.Markdown(
            "Find books endorsed by people whose taste you trust. "
            "Searches back-cover blurbs, forewords, jacket quotes, and praise pages."
        )
        gr.Markdown(
            "_First search takes 30–90 seconds; instant on repeat thanks to caching._"
        )

        person_name = gr.Textbox(
            label="Name",
            placeholder="e.g. John Carreyrou, Bradley Hope, Michael Bloomberg",
        )
        casual_toggle = gr.Checkbox(
            label="Also show casual shares (tweets, blog posts, podcasts) when no blurbs found",
            value=False,
        )
        name_btn = gr.Button("Find Books", variant="primary")
        name_output = gr.Markdown()
        with gr.Accordion("Process log", open=False):
            name_log = gr.Textbox(label="", lines=15)
        name_btn.click(
            run_search,
            inputs=[person_name, casual_toggle],
            outputs=[name_output, name_log],
        )
        person_name.submit(
            run_search,
            inputs=[person_name, casual_toggle],
            outputs=[name_output, name_log],
        )

    pw_btn.click(check_password, inputs=[pw_input], outputs=[login_section, app_section, pw_error])
    pw_input.submit(check_password, inputs=[pw_input], outputs=[login_section, app_section, pw_error])


if __name__ == "__main__":
    if not APP_PASSWORD:
        print("WARNING: APP_PASSWORD not set. App is unprotected.")
    else:
        print(f"App protected. Password length: {len(APP_PASSWORD)}")
    demo.launch(ssr_mode=False)
"""
next-read: web app interface
"""

import os
from urllib.parse import quote_plus
import requests
import gradio as gr
from dotenv import load_dotenv
from next_read import recommend_from_book, recommend_from_name

load_dotenv()

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")


def _normalize_name(name: str) -> str:
    """Collapse whitespace and title-case so 'michael  bloomberg' -> 'Michael Bloomberg'."""
    return " ".join(name.strip().split()).title()


def _normalize_title(text: str) -> str:
    """Just collapse whitespace; preserve user's casing for titles."""
    return " ".join(text.strip().split())


def check_password(pw):
    if pw == APP_PASSWORD and APP_PASSWORD:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "",
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        "Wrong password. Try again.",
    )


def run_book_mode(title, author):
    title = _normalize_title(title)
    author = _normalize_name(author)
    if not title or not author:
        return "Please enter both a book title and author.", ""
    logs = []
    def logger(msg):
        logs.append(str(msg))
    try:
        result = recommend_from_book(title, author, log=logger)
        return _format_output(result), "\n".join(logs)
    except Exception as e:
        return f"Error: {e}", "\n".join(logs)


def run_name_mode(name):
    name = _normalize_name(name)
    if not name:
        return "Please enter a person's name.", ""
    logs = []
    def logger(msg):
        logs.append(str(msg))
    try:
        result = recommend_from_name(name, log=logger)
        return _format_output(result), "\n".join(logs)
    except Exception as e:
        return f"Error: {e}", "\n".join(logs)


def _get_isbn(title, author):
    """Look up ISBN for a book via Google Books API. Returns None if not found."""
    try:
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
    query = quote_plus(f"{title} {author}")
    return f"https://www.amazon.com/s?k={query}&i=stripbooks"


def _format_output(result):
    recs = result.get("recommendations", [])
    lines = []
    if result.get("input_book"):
        b = result["input_book"]
        lines.append(f"# Recommendations based on *{b['title']}* by {b['author']}\n")
        blurbers = result.get("blurbers", [])
        if blurbers:
            names = ", ".join(blurber["name"] for blurber in blurbers if blurber.get("name"))
            lines.append(f"**Blurbers found:** {names}\n")
    elif result.get("input_name"):
        lines.append(f"# Books endorsed by {result['input_name']}\n")
    if not recs:
        lines.append("\n_No verified recommendations found. Try a different name or book, or check the process log below for details._")
        return "\n".join(lines)
    for i, rec in enumerate(recs, 1):
        endorsers = ", ".join(rec["endorsers"])
        year = rec.get("year") or "n/a"
        amazon_url = _amazon_link(rec["title"], rec["author"])
        lines.append(f"### {i}. *{rec['title']}* ({year})")
        # HTML anchor for the Amazon link (Gradio markdown was mangling
        # plain-markdown links inside this line). target=_blank so it opens
        # in a new tab and doesn't navigate away from the app.
        lines.append(
            f"by {rec['author']}  \u00b7  "
            f"<a href='{amazon_url}' target='_blank' rel='noopener'>Buy on Amazon</a>"
        )
        lines.append(f"**Endorsed by:** {endorsers}")
        if rec.get("one_line"):
            lines.append(f"_{rec['one_line']}_")
        for ev in rec.get("evidence", []):
            if ev.get("quote"):
                quote = ev["quote"]
                if len(quote) > 200:
                    quote = quote[:200] + "..."
                lines.append(f"> \"{quote}\" \u2014 {ev['endorser']}")
        lines.append("")
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
        gr.Markdown("Find books endorsed by people whose taste you trust.")

        with gr.Tab("By Book"):
            gr.Markdown(
                "Enter a book. We find the blurbers on the back cover, "
                "then find what those people endorsed elsewhere. "
                "(First search takes 3-6 minutes; instant after that thanks to caching.)"
            )
            with gr.Row():
                book_title = gr.Textbox(label="Book Title", placeholder="e.g. Streetwise: Getting to and Through Goldman Sachs")
                book_author = gr.Textbox(label="Author", placeholder="e.g. Lloyd Blankfein")
            book_btn = gr.Button("Find Recommendations", variant="primary")
            book_output = gr.Markdown()
            with gr.Accordion("Process log", open=False):
                book_log = gr.Textbox(label="", lines=15)
            book_btn.click(run_book_mode, inputs=[book_title, book_author], outputs=[book_output, book_log])

        with gr.Tab("By Person"):
            gr.Markdown(
                "Enter the name of an endorser. We find every book they've blurbed. "
                "(First search takes 2-4 minutes; instant after that thanks to caching.)"
            )
            person_name = gr.Textbox(label="Endorser Name", placeholder="e.g. Michael Bloomberg")
            name_btn = gr.Button("Find Endorsements", variant="primary")
            name_output = gr.Markdown()
            with gr.Accordion("Process log", open=False):
                name_log = gr.Textbox(label="", lines=15)
            name_btn.click(run_name_mode, inputs=[person_name], outputs=[name_output, name_log])

    pw_btn.click(
        check_password,
        inputs=[pw_input],
        outputs=[login_section, app_section, pw_error],
    )
    pw_input.submit(
        check_password,
        inputs=[pw_input],
        outputs=[login_section, app_section, pw_error],
    )


if __name__ == "__main__":
    if not APP_PASSWORD:
        print("WARNING: APP_PASSWORD not set. App is unprotected.")
    else:
        print(f"App protected. Password length: {len(APP_PASSWORD)}")
    demo.launch(ssr_mode=False)
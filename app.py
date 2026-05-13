"""
next-read: web app interface
Run: python app.py
Then open http://localhost:7860 in your browser.

Auth: set APP_USERNAME and APP_PASSWORD in .env to enable login.
If either is missing, the app launches without auth.
"""

import os
import gradio as gr
from dotenv import load_dotenv
from next_read import recommend_from_book, recommend_from_name

load_dotenv()


def run_book_mode(title, author):
    if not title.strip() or not author.strip():
        return "Please enter both a book title and author.", ""
    
    logs = []
    def logger(msg):
        logs.append(str(msg))
    
    try:
        result = recommend_from_book(title.strip(), author.strip(), log=logger)
        return _format_output(result), "\n".join(logs)
    except Exception as e:
        return f"Error: {e}", "\n".join(logs)


def run_name_mode(name):
    if not name.strip():
        return "Please enter a person's name.", ""
    
    logs = []
    def logger(msg):
        logs.append(str(msg))
    
    try:
        result = recommend_from_name(name.strip(), log=logger)
        return _format_output(result), "\n".join(logs)
    except Exception as e:
        return f"Error: {e}", "\n".join(logs)


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
        lines.append(f"### {i}. *{rec['title']}* ({year})")
        lines.append(f"by {rec['author']}")
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
    gr.Markdown("# NextRead")
    gr.Markdown("Find books endorsed by people whose taste you trust.")
    
    with gr.Tab("By Book"):
        gr.Markdown("Enter a book. We find the blurbers on the back cover, then find what those people endorsed elsewhere. (Takes 1-2 minutes.)")
        with gr.Row():
            book_title = gr.Textbox(label="Book Title", placeholder="e.g. Streetwise: Getting to and Through Goldman Sachs")
            book_author = gr.Textbox(label="Author", placeholder="e.g. Lloyd Blankfein")
        book_btn = gr.Button("Find Recommendations", variant="primary")
        book_output = gr.Markdown()
        with gr.Accordion("Process log", open=False):
            book_log = gr.Textbox(label="", lines=15)
        book_btn.click(run_book_mode, inputs=[book_title, book_author], outputs=[book_output, book_log])
    
    with gr.Tab("By Person"):
        gr.Markdown("Enter the name of an endorser. We find every book they've blurbed. (Takes 30-60 seconds.)")
        person_name = gr.Textbox(label="Endorser Name", placeholder="e.g. Michael Bloomberg")
        name_btn = gr.Button("Find Endorsements", variant="primary")
        name_output = gr.Markdown()
        with gr.Accordion("Process log", open=False):
            name_log = gr.Textbox(label="", lines=15)
        name_btn.click(run_name_mode, inputs=[person_name], outputs=[name_output, name_log])


if __name__ == "__main__":
    username = os.environ.get("APP_USERNAME")
    password = os.environ.get("APP_PASSWORD")
    
    if username and password:
        print(f"Launching with auth (username: {repr(username)}, password length: {len(password)})")
        demo.launch(auth=(username, password), ssr_mode=False)
    else:
        print("Launching without auth (set APP_USERNAME and APP_PASSWORD in .env to enable login)")
        demo.launch(ssr_mode=False)
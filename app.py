"""AfyaPlus Multimodal Intake Assistant - Gradio app (Deliverable 4).

One upload box accepts either a photo or a voice note. router.py decides
which pipeline handles it; this file only renders the result in
plain, non-technical language for front-desk / community health workers.

Run: python app.py, then open the printed local URL.
"""
import gradio as gr

from router import route

APP_TITLE = "AfyaPlus Intake Assistant (Demo)"

INTRO_MD = """
# AfyaPlus Intake Assistant (Demo)

Upload **one photo** of a skin concern, or **one voice note** describing symptoms.
The assistant will write a short, plain-language note for the clinical team to review.

**This tool does not diagnose anything.** Every note it writes must be checked by a
qualified health worker before any care decision is made.
"""


def handle_upload(file_obj):
    """Gradio click-handler wired to submit_btn below. file_obj is whatever
    Gradio's gr.File component hands back for the uploaded file (a path
    string in some Gradio versions, an object with a .name path in others --
    the isinstance check below handles both). This function's only job is
    formatting: all the actual pipeline logic lives in router.route(); this
    just turns its result dict into the two Markdown/text outputs the UI
    displays."""
    if file_obj is None:
        return "Please upload a photo or a voice note to continue.", ""

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    outcome = route(file_path)
    modality = outcome["modality"]
    result = outcome["result"]

    if modality == "image":
        if result["caption"] is None:
            # caption_image() returned None -- either UNREADABLE or
            # OUT_OF_SCOPE; result["note"] explains which and why.
            body = f"**Could not generate a note.** {result['note']}"
        else:
            body = f"**AI-written note:** {result['caption']}"
        flags = ", ".join(result["flags"]) if result["flags"] else "none"
        summary = (
            f"### Photo submission\n\n{body}\n\n"
            f"**Review flags:** {flags}\n\n"
            f"**Safety note:** {result['disclaimer']}"
        )
        return summary, ""  # second output (raw_transcript box) is audio-only, left blank

    # audio: build a bullet list of the 5 structured fields, showing
    # "not mentioned" for any field extract_fields() left as None rather
    # than silently omitting it, so the clinician sees exactly what wasn't
    # captured.
    fields = result["fields"]
    fields_md = "\n".join(f"- **{k.capitalize()}:** {v or 'not mentioned'}" for k, v in fields.items())
    summary = (
        f"### Voice note submission\n\n"
        f"**Detected language:** {result['language']} "
        f"(confidence {result['language_probability']:.0%})\n\n"
        f"**Transcript:** {result['text']}\n\n"
        f"**Structured intake fields:**\n{fields_md}\n\n"
        f"**Safety note:** {result['disclaimer']}"
    )
    return summary, result["text"]


# Single-page Gradio Blocks layout: one file input accepts either modality
# (file_types=["image", "audio"] lets Gradio's picker show both), one button
# triggers handle_upload(), and two outputs receive its two return values --
# output_md (the rendered Markdown note) and raw_transcript (kept hidden;
# holds the plain-text transcript for audio submissions, in case a caller
# wants the unformatted text rather than the Markdown summary).
with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(INTRO_MD)
    with gr.Row():
        file_input = gr.File(
            label="Photo or voice note",
            file_types=["image", "audio"],
        )
    submit_btn = gr.Button("Get AI note for clinical review", variant="primary")
    output_md = gr.Markdown(label="Result")
    raw_transcript = gr.Textbox(label="Raw transcript (audio only)", visible=False)

    # Wire the button: on click, run handle_upload(file_input's value) and
    # route its (summary, transcript) tuple to the two output components.
    submit_btn.click(fn=handle_upload, inputs=file_input, outputs=[output_md, raw_transcript])

if __name__ == "__main__":
    demo.launch()  # prints a local URL (default http://127.0.0.1:7860) to open in a browser

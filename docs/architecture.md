# Architecture

This document describes how the AfyaPlus Multimodal Intake Assistant is
put together: the runtime request path, the offline retrieval-index build,
the internal decision logic inside captioning/transcription, and the
optional API-comparison path. See [`../README.md`](../README.md) for setup
and the top-level pipeline summary, and [`../config.py`](../config.py) for
every constant referenced below.

## 1. Component overview

Everything funnels through one router and one Gradio app. `config.py` is
the single source of truth for paths, model names, thresholds, and
disclaimer text -- every module below imports it rather than hardcoding
any of those values.

```mermaid
flowchart TB
    subgraph UI["Presentation"]
        App["app.py<br/>Gradio Blocks UI"]
    end

    subgraph Dispatch["Dispatch"]
        Router["router.py<br/>classify_input() / route()"]
    end

    subgraph ImagePipeline["Image pipeline (Deliverable 2)"]
        Caption["caption_image.py<br/>caption_image()"]
        Unreadable["_looks_unreadable()<br/>grayscale std-dev check"]
        OOS["_is_out_of_scope()<br/>CLIP zero-shot vs DOMAIN_LABELS"]
        BLIP["BLIP-base<br/>Salesforce/blip-image-captioning-base"]
    end

    subgraph AudioPipeline["Audio pipeline (Deliverable 3)"]
        Transcribe["transcribe_audio.py<br/>transcribe() / transcribe_long()"]
        Whisper["faster-whisper<br/>base, CTranslate2, int8"]
        Extract["extract_fields.py<br/>extract_fields()"]
        LLMExtract["flan-t5-small<br/>_llm_extract()"]
        RegexExtract["regex safety-net<br/>_regex_fallback()"]
    end

    subgraph RetrievalPipeline["Retrieval pipeline (Deliverable 1)"]
        BuildIndex["build_index.py<br/>offline, run once"]
        Search["search.py<br/>search() / domain_similarity()"]
        CLIP["CLIP<br/>openai/clip-vit-base-patch32"]
        IndexFile[("outputs/clip_index.npz<br/>filenames + embeddings")]
    end

    Config[["config.py<br/>paths, thresholds, disclaimers"]]

    App -->|"file_obj"| Router
    Router -->|"image/*"| Caption
    Router -->|"audio/*"| Transcribe

    Caption --> Unreadable
    Unreadable -->|"passes"| OOS
    OOS -->|"in-scope"| BLIP
    OOS -.->|"uses"| Search
    BLIP --> Caption

    Transcribe --> Whisper
    Whisper --> Extract
    Extract --> LLMExtract
    Extract --> RegexExtract
    LLMExtract -.->|"merged, LLM value wins if present"| Extract
    RegexExtract -.->|"else fallback fills gaps"| Extract

    BuildIndex --> CLIP
    CLIP --> IndexFile
    Search --> IndexFile
    Search --> CLIP

    Config -.-> Caption
    Config -.-> Transcribe
    Config -.-> Extract
    Config -.-> Search
    Config -.-> Router

    Caption -->|"caption, flags, disclaimer"| Router
    Transcribe -->|"transcript, fields, disclaimer"| Router
    Router -->|"uniform result envelope"| App
```

## 2. Request flow: photo submission

```mermaid
sequenceDiagram
    actor User
    participant App as app.py
    participant Router as router.py
    participant Cap as caption_image.py
    participant Srch as search.py (domain_similarity)
    participant BLIP as BLIP-base model

    User->>App: upload photo, click "Get AI note"
    App->>Router: route(file_path)
    Router->>Router: classify_input() -> "image"
    Router->>Cap: caption_image(file_path)

    Cap->>Cap: _looks_unreadable(image)<br/>grayscale std-dev < 12.0?
    alt image unreadable
        Cap-->>Router: {caption: null,<br/>flags: [UNREADABLE, HUMAN_REVIEW_REQUIRED]}
    else image readable
        Cap->>Srch: domain_similarity(image)
        Srch->>Srch: CLIP image embedding vs<br/>12 zero-shot DOMAIN_LABELS
        Srch-->>Cap: best similarity score
        alt score < OUT_OF_SCOPE_SIMILARITY_CEILING (0.24)
            Cap-->>Router: {caption: null,<br/>flags: [OUT_OF_SCOPE, HUMAN_REVIEW_REQUIRED]}
        else in scope
            Cap->>BLIP: generate("a medical photo of", image)
            BLIP-->>Cap: caption text
            Cap-->>Router: {caption: text,<br/>flags: [HUMAN_REVIEW_REQUIRED]}
        end
    end

    Router-->>App: {modality: "image", result: {...}}
    App->>App: render caption/flags/disclaimer as Markdown
    App-->>User: plain-language note for clinician review
```

Every branch -- unreadable, out-of-scope, or a successful caption -- always
carries `config.CAPTION_DISCLAIMER` and always sets
`HUMAN_REVIEW_REQUIRED`; nothing is ever shown without both.

## 3. Request flow: voice note submission

```mermaid
sequenceDiagram
    actor User
    participant App as app.py
    participant Router as router.py
    participant Tr as transcribe_audio.py
    participant WM as faster-whisper (WhisperModel)
    participant Ex as extract_fields.py
    participant T5 as flan-t5-small

    User->>App: upload voice note, click "Get AI note"
    App->>Router: route(file_path)
    Router->>Router: classify_input() -> "audio"
    Router->>Tr: transcribe(file_path)

    Tr->>WM: transcribe(audio, vad_filter=True,<br/>min_silence_duration_ms=500)
    WM-->>Tr: segments, detected language + confidence
    Tr->>Tr: join segment text -> full transcript

    Tr->>Ex: extract_fields(transcript)
    par LLM path
        Ex->>T5: generate(prompt with transcript)
        T5-->>Ex: raw text -> parsed JSON (best-effort)
    and regex safety-net
        Ex->>Ex: _regex_fallback(transcript)<br/>symptom aliases, locations,<br/>duration regex, severity words,<br/>_detect_fever() negation check
    end
    Ex->>Ex: merge: llm_result[k] or fallback[k]<br/>for each of 5 fields
    Ex-->>Tr: {symptom, location, duration,<br/>severity, fever}

    Tr-->>Router: {text, language, fields,<br/>disclaimer}
    Router-->>App: {modality: "audio", result: {...}}
    App->>App: render transcript + structured<br/>fields + disclaimer as Markdown
    App-->>User: plain-language note for clinician review
```

`_detect_fever()` specifically guards against a real bug class: a naive
`"fever" in text` check would flip "I haven't had a fever" to a positive
fever flag. It scans a window before the word "fever" for negation cues
(`no`, `not`, `n't`, `without`, `denies`) before deciding yes/no.

## 4. Retrieval: offline index build vs. query time

```mermaid
flowchart LR
    subgraph Offline["Offline, run once: build_index.py"]
        Images[("data/images/index/<br/>17 PNGs")]
        CLIP1["CLIP image encoder"]
        Norm1["L2-normalize embeddings"]
        Save[("outputs/clip_index.npz<br/>filenames[] + embeddings[17x512]")]
        Images --> CLIP1 --> Norm1 --> Save
    end

    subgraph QueryTime["Query time: search.py"]
        Query["text query,<br/>e.g. 'a red itchy rash'"]
        CLIP2["CLIP text encoder"]
        Norm2["L2-normalize"]
        Dot["dot product vs all<br/>stored embeddings<br/>(cosine similarity, unit vectors)"]
        Rank["argsort descending,<br/>take top_k"]
        Thresh{"similarity >=<br/>MIN_SIMILARITY_THRESHOLD<br/>(0.19)?"}
        Query --> CLIP2 --> Norm2 --> Dot
        Save -.->|"loaded at query time"| Dot
        Dot --> Rank --> Thresh
        Thresh -->|"yes"| Above["above_threshold: true"]
        Thresh -->|"no"| Below["above_threshold: false<br/>(shown but flagged low-confidence)"]
    end
```

`MIN_SIMILARITY_THRESHOLD` (0.19) and `OUT_OF_SCOPE_SIMILARITY_CEILING`
(0.24, used by the captioning pipeline's `domain_similarity` check) are
both empirically calibrated by `python search.py --calibrate` against this
specific 17-image index -- see the docstring comments in
[`../config.py`](../config.py) for the exact relevant/irrelevant score
spread that justified each number. Both numbers are index-dependent and
should be re-derived if the index images change.

## 5. Evaluation and the optional API-comparison path

```mermaid
flowchart TB
    Eval["evaluate.py"]
    Compare["compare_api_vs_local.py<br/>(optional, spends OpenAI credit)"]

    Eval -->|"search(), caption_image() x3,<br/>transcribe() x2"| LocalOnly["local pipeline only"]
    LocalOnly --> RawJSON[("outputs/evaluation_raw.json")]
    LocalOnly --> EvalTable[("docs/evaluation_table.md")]

    Compare -->|"same 3 images + 2 audio"| LocalPath["local pipeline<br/>(caption_image, transcribe)"]
    Compare -->|"same 3 images + 2 audio"| APIPath["OpenAI API<br/>GPT-4o-mini vision +<br/>Whisper API"]
    LocalPath --> CompareTable[("docs/api_vs_local_comparison.md")]
    APIPath --> CompareTable

    EvalTable -.->|"referenced by"| Memo[("docs/business_memo.md")]
    CompareTable -.->|"referenced by"| Memo
```

`evaluate.py` is part of the normal, no-API-key-needed workflow.
`compare_api_vs_local.py` is a separate, explicitly opt-in script (see
README "API vs local fallback") that requires `OPENAI_API_KEY` in `.env`
and is not on the path of any other script -- nothing else imports it or
depends on its output.

## 6. Design principles this diagram is enforcing

- **Single source of config.** Every threshold, path, and disclaimer
  string lives in [`../config.py`](../config.py); no module hardcodes its
  own copy, so calibration stays in one place.
- **Fail toward a human, not a guess.** Both pipelines are structured so
  every early-exit branch (`UNREADABLE`, `OUT_OF_SCOPE`) and every
  successful branch alike ends with `HUMAN_REVIEW_REQUIRED` plus a fixed
  disclaimer -- there is no code path that returns a caption or transcript
  without both.
- **LLM output is never trusted alone.** `extract_fields.py` always runs
  the regex fallback alongside the flan-t5-small call and merges field-by-
  field, so a field the model hallucinates away (or gets wrong) can still
  be recovered from the raw transcript text.
- **Local is the default; API is opt-in and isolated.** Only
  `compare_api_vs_local.py` touches the network/OpenAI; every other script
  in the request path (`router.py`, `app.py`, `evaluate.py`) is fully
  local/offline after the one-time Hugging Face model download.

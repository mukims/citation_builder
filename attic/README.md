# Attic

Things kept for reference that are **not part of the running pipeline**. Nothing
in the citation pipeline imports, reads, or executes anything in here.

Kept rather than deleted because most of it is either expensive to recreate or
worth a second look before it goes for good. Delete freely once you've decided.

| Item | Why it's here |
|------|---------------|
| `agent_graph.py` | The LangGraph ReAct supervisor. The orchestrator used it for the `raw/` and `drafts/` events, but both are fixed sequences, so they now call the agents directly. Nothing imports this any more. Its tool wrappers also returned success strings unconditionally — `batch_cite_tool` reported a written draft even when Agent 5 had aborted without writing one. |
| `publaynet_config.yaml` | A local PubLayNet model config. `config.py` uses `DETECTRON_CONFIG = "lp://PubLayNet/..."`, which layoutparser downloads, so this file is never read. |
| `publaynet_model_final.pth` | **Byte-identical duplicate** of `model_final.pth` in the repo root (same MD5). `config.DETECTRON_WEIGHTS` points at the root copy. Deleting this reclaims 817 MB. |
| `detectron2/` | A source checkout of detectron2. The package is installed properly in site-packages (`detectron2-0.6.dist-info`), not from here, so this is 21 MB of reference material. |
| `grobid_output/` | 167 TEI XML files plus text dumps from a GROBID run. Nothing in the codebase reads them, and their `<email>` elements carry author addresses lifted from paper front pages. |
| `citation_builder_agent/` | A Google ADK scratch experiment: a stub `algebra()` tool and a model name (`gemma4:e2b`) that doesn't exist. Wired into nothing. It also broke `unittest discover` from the repo root, since discovery walked into it and failed on the missing `google.adk` import. |
| `text_extraction.ipynb` | Notebook used to produce `extracted_metadata.jsonl`. Superseded by the GROBID path; kept because it documents how that data was made. |
| `archive_pulled_pdfs/` | A vestigial snapshot directory. Only ever held a `.gitkeep`; nothing references it. |

## Restoring something

`agent_graph.py`, `publaynet_config.yaml` and `archive_pulled_pdfs/.gitkeep` are
still tracked, so their history moved with them and `git mv` puts them back.
Everything else is local-only and ignored by git.

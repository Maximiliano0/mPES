"""Export new documentation markdown files to HTML."""

import markdown
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.parent  # workspace root

DOCS = [
    "doc/comparacion_modelos.md",
    "tabular/pes_ql/doc/pes_ql_explained.md",
    "tabular/pes_ql/doc/pes_ql_theory.md",
    "tabular/pes_dql/doc/pes_dql_explained.md",
    "tabular/pes_dql/doc/pes_dql_theory.md",
    "ml/pes_dqn/doc/pes_dqn_explained.md",
    "ml/pes_dqn/doc/pes_dqn_theory.md",
    "ml/pes_rdqn/doc/pes_rdqn_explained.md",
    "ml/pes_rdqn/doc/pes_rdqn_theory.md",
    "ml/pes_a2c/doc/pes_a2c_explained.md",
    "ml/pes_a2c/doc/pes_a2c_theory.md",
    "ml/pes_trf/doc/pes_trf_explained.md",
    "ml/pes_trf/doc/pes_trf_theory.md",
    "ml/pes_ens/doc/pes_ens_explained.md",
    "ml/pes_ens/doc/pes_ens_theory.md",
]

HTML_TEMPLATE = (
    "<!DOCTYPE html>\n"
    "<html lang=\"es\">\n"
    "<head>\n"
    "<meta charset=\"UTF-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<title>{title}</title>\n"
    "<style>\n"
    "  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;"
    " max-width: 960px; margin: 0 auto; padding: 2rem; color: #222; line-height: 1.6; }}\n"
    "  h1,h2,h3 {{ color: #1a1a2e; }}\n"
    "  code {{ background: #f4f4f4; padding: .2em .4em; border-radius: 3px; font-size: .9em; }}\n"
    "  pre code {{ display: block; padding: 1em; overflow-x: auto; }}\n"
    "  table {{ border-collapse: collapse; width: 100%; }}\n"
    "  th,td {{ border: 1px solid #ccc; padding: .5em 1em; }}\n"
    "  th {{ background: #f0f0f0; }}\n"
    "  blockquote {{ border-left: 4px solid #ccc; margin: 0; padding-left: 1em; color: #555; }}\n"
    "</style>\n"
    "<script src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>\n"
    "</head>\n"
    "<body>\n"
    "{body}\n"
    "</body>\n"
    "</html>"
)

md_parser = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])

for doc_rel in DOCS:
    p = ROOT / doc_rel
    if not p.exists():
        print(f"MISSING: {doc_rel}")
        continue
    text = p.read_text(encoding="utf-8")
    md_parser.reset()
    body = md_parser.convert(text)
    title = p.stem.replace("_", " ").title()
    html = HTML_TEMPLATE.format(title=title, body=body)
    out = p.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    print(f"OK: {out.relative_to(ROOT)}")

print("Done.")

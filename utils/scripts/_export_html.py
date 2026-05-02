"""
utils/_export_html.py — Convert Markdown documentation to HTML.

Converts every ``.md`` file inside a package's ``doc/`` directory to a
matching ``.html`` in the same location, using the project's standard
template (KaTeX math + dark-mode CSS).

Usage
-----
    # Convert a single package
    python utils/_export_html.py pes_dqn

    # Convert several packages at once
    python utils/_export_html.py pes_a2c pes_dqn pes_trf

    # Convert ALL packages (no arguments)
    python utils/_export_html.py
"""

##########################
##  Imports externos    ##
##########################
import os
import sys
import glob
import markdown


##########################
##  Constantes          ##
##########################

# Packages that contain a doc/ directory.
ALL_PACKAGES = ["pes_base", "pes_ql", "pes_dql", "pes_dqn", "pes_rdqn", "pes_a2c", "pes_trf", "pes_ens"]

# Workspace root: two levels up from this script (utils/scripts/_export_html.py).
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
  <script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
  <script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{delimiters:[
          {{left:'$$',right:'$$',display:true}},
          {{left:'$',right:'$',display:false}}
        ]}})"></script>
  <style>
    :root {{ --bg:#ffffff; --fg:#1a1a2e; --accent:#0f3460; --code-bg:#f4f4f8;
             --border:#ddd; --link:#1565c0; --table-stripe:#f9f9fc; }}
    @media (prefers-color-scheme:dark) {{
      :root {{ --bg:#1a1a2e; --fg:#e0e0e0; --accent:#64b5f6; --code-bg:#16213e;
               --border:#333; --link:#90caf9; --table-stripe:#1e2a45; }}
    }}
    *,*::before,*::after {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ font-family:"Segoe UI",system-ui,-apple-system,sans-serif;
           line-height:1.7; color:var(--fg); background:var(--bg);
           max-width:52em; margin:2em auto; padding:0 1.5em; }}
    h1 {{ border-bottom:3px solid var(--accent); padding-bottom:.3em; }}
    h2 {{ border-bottom:1px solid var(--border); padding-bottom:.2em; margin-top:2em; }}
    h3,h4 {{ margin-top:1.6em; }}
    a {{ color:var(--link); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    code {{ font-family:"JetBrains Mono","Fira Code",Consolas,monospace;
           font-size:.92em; background:var(--code-bg); padding:.15em .35em;
           border-radius:4px; }}
    pre {{ background:var(--code-bg); padding:1em 1.2em; border-radius:8px;
          overflow-x:auto; border:1px solid var(--border); }}
    pre code {{ background:none; padding:0; font-size:.88em; }}
    table {{ border-collapse:collapse; width:100%; margin:1em 0; }}
    th,td {{ border:1px solid var(--border); padding:.55em .9em; text-align:left; }}
    th {{ background:var(--accent); color:#fff; }}
    tr:nth-child(even) {{ background:var(--table-stripe); }}
    blockquote {{ border-left:4px solid var(--accent); margin-left:0;
                 padding:.4em 1em; color:#666; background:var(--code-bg);
                 border-radius:0 6px 6px 0; }}
    .katex-display {{ overflow-x:auto; overflow-y:hidden; padding:.5em 0; }}
  </style>
</head>
<body>
  {body}
</body>
</html>'''

MARKDOWN_EXTENSIONS = ["tables", "fenced_code", "toc"]


##########################
##  Funciones           ##
##########################

def _extract_title(md_text):
    """Extract the first ``# heading`` from Markdown text as HTML title.

    Parameters
    ----------
    md_text : str
        Raw Markdown content.

    Returns
    -------
    str
        The heading text (without ``#``), or ``"Documentation"`` if none found.
    """
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            return stripped.lstrip("# ").strip()
    return "Documentation"


def convert_md_to_html(md_path, html_path):
    """Convert a single Markdown file to HTML using the project template.

    Parameters
    ----------
    md_path : str
        Absolute path to the source ``.md`` file.
    html_path : str
        Absolute path for the output ``.html`` file.
    """
    with open(md_path, "r", encoding="utf-8") as fh:
        md_text = fh.read()

    title = _extract_title(md_text)
    body = markdown.markdown(md_text, extensions=MARKDOWN_EXTENSIONS)
    html = HTML_TEMPLATE.format(title=title, body=body)

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def export_package(pkg_name):
    """Convert every ``.md`` inside ``<pkg>/doc/`` to ``.html``.

    Parameters
    ----------
    pkg_name : str
        Package directory name (e.g. ``"pes_dqn"``).

    Returns
    -------
    int
        Number of files converted.
    """
    doc_dir = os.path.join(WORKSPACE_ROOT, pkg_name, "doc")
    if not os.path.isdir(doc_dir):
        print(f"  SKIP {pkg_name}/doc/ — directory not found")
        return 0

    md_files = sorted(glob.glob(os.path.join(doc_dir, "*.md")))
    if not md_files:
        print(f"  SKIP {pkg_name}/doc/ — no .md files")
        return 0

    count = 0
    for md_path in md_files:
        html_path = os.path.splitext(md_path)[0] + ".html"
        convert_md_to_html(md_path, html_path)
        rel = os.path.relpath(html_path, WORKSPACE_ROOT)
        print(f"  OK  {rel}")
        count += 1

    return count


##########################
##  Entry point         ##
##########################

if __name__ == "__main__":
    packages = sys.argv[1:] if len(sys.argv) > 1 else ALL_PACKAGES

    total = 0
    for pkg in packages:
        total += export_package(pkg)

    print(f"\nDone — {total} file(s) converted.")

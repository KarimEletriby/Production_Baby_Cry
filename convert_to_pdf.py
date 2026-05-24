import markdown
import re
import os
import subprocess
import time

md_path = "C:\\Users\\CRIZMA\\.gemini\\antigravity-ide\\brain\\e44386a7-40fd-44ff-8e96-82d464ee1d60\\technical_report.md"
html_path = "C:\\Users\\CRIZMA\\.gemini\\antigravity-ide\\brain\\e44386a7-40fd-44ff-8e96-82d464ee1d60\\technical_report.html"
pdf_path = "c:\\BabyCry\\BabyCry_Technical_Report.pdf"

with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Python markdown does not automatically convert mermaid code blocks to what mermaid expects unless we use extensions.
# So let's replace ```mermaid ... ``` with <pre class="mermaid"> ... </pre>
def mermaid_replacer(match):
    code = match.group(1)
    return f'<pre class="mermaid">{code}</pre>'

text = re.sub(r'```mermaid\n(.*?)\n```', mermaid_replacer, text, flags=re.DOTALL)

html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Baby Cry Technical Report</title>
<style>
body {{
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
    line-height: 1.6;
    color: #24292e;
    max-width: 900px;
    margin: 0 auto;
    padding: 30px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 20px;
}}
table, th, td {{
    border: 1px solid #dfe2e5;
}}
th, td {{
    padding: 8px 13px;
}}
th {{
    background-color: #f6f8fa;
    font-weight: 600;
}}
pre {{
    background-color: #f6f8fa;
    border-radius: 3px;
    padding: 16px;
    overflow: auto;
}}
code {{
    background-color: rgba(27,31,35,0.05);
    border-radius: 3px;
    padding: 0.2em 0.4em;
    font-family: "SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
}}
pre code {{
    background-color: transparent;
    padding: 0;
}}
h1, h2, h3 {{
    border-bottom: 1px solid #eaecef;
    padding-bottom: 0.3em;
}}
.mermaid {{
    text-align: center;
    margin: 20px 0;
}}
</style>
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true }});
</script>
</head>
<body>
{html_content}
</body>
</html>
"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(full_html)

print("Generated HTML file. Now converting to PDF using Edge...")

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
# Edge needs full URI
html_uri = "file:///" + html_path.replace("\\", "/")

cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--print-to-pdf=" + pdf_path,
    "--virtual-time-budget=10000",
    "--no-pdf-header-footer",
    html_uri
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd)

if os.path.exists(pdf_path):
    print("PDF generated successfully at:", pdf_path)
else:
    print("PDF generation failed.")

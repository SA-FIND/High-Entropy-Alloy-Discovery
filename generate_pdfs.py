#!/usr/bin/env python3
"""
generate_pdfs.py — Generates publication-grade academic PDFs for:
1. RESEARCH_PROPOSAL.md -> Solomon_Ahedor_Research_Proposal_MetaForge.pdf
2. COMPREHENSIVE_HEA_COURSE.md -> Solomon_Ahedor_HEA_Master_Foundations_Course.pdf

Uses Headless Microsoft Edge for pixel-perfect print rendering with A4 pagination.
"""

import os
import re
import html
import subprocess
import time

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSS_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-right {
        content: counter(page);
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #64748b;
    }
}

* {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.62;
    font-size: 9.8pt;
    margin: 0;
    padding: 0;
}

/* ═══════════════════════════════════════════════
   COVER / HEADER BANNER
   ═══════════════════════════════════════════════ */
.doc-header {
    border-bottom: 2.5px solid #0f172a;
    padding-bottom: 20px;
    margin-bottom: 24px;
}

.doc-badge {
    display: inline-block;
    background: #0f172a;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.5pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    padding: 3px 9px;
    border-radius: 4px;
    margin-bottom: 12px;
}

.doc-title {
    font-family: 'Inter', sans-serif;
    font-size: 19pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.22;
    margin: 0 0 12px 0;
    letter-spacing: -0.02em;
}

.meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 8.8pt;
    margin-top: 14px;
}

.meta-item strong {
    color: #0f172a;
    font-weight: 600;
    display: inline-block;
    margin-right: 4px;
}

/* ═══════════════════════════════════════════════
   HEADINGS & TYPOGRAPHY
   ═══════════════════════════════════════════════ */
h1 {
    font-family: 'Inter', sans-serif;
    font-size: 13.5pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 1.5px solid #0284c7;
    padding-bottom: 5px;
    margin-top: 26px;
    margin-bottom: 12px;
    page-break-after: avoid;
}

h2 {
    font-family: 'Inter', sans-serif;
    font-size: 11.5pt;
    font-weight: 700;
    color: #0369a1;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

h3 {
    font-size: 10.2pt;
    font-weight: 600;
    color: #334155;
    margin-top: 15px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

p {
    margin: 0 0 10px 0;
    text-align: justify;
}

/* ═══════════════════════════════════════════════
   CALLOUTS & BLOCKQUOTES
   ═══════════════════════════════════════════════ */
blockquote {
    margin: 14px 0;
    padding: 12px 16px;
    background: #f0f9ff;
    border-left: 4px solid #0284c7;
    border-radius: 0 6px 6px 0;
    font-size: 9.3pt;
    color: #0c4a6e;
    page-break-inside: avoid;
}

blockquote p {
    margin: 0 0 6px 0;
}

blockquote p:last-child {
    margin: 0;
}

.qa-card {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #0284c7;
    border-radius: 6px;
    padding: 12px 15px;
    margin: 14px 0;
    page-break-inside: avoid;
}

.qa-question {
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
    font-size: 9.8pt;
}

.qa-answer {
    color: #334155;
    font-size: 9.3pt;
    line-height: 1.55;
}

/* ═══════════════════════════════════════════════
   TABLES
   ═══════════════════════════════════════════════ */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 8.8pt;
    page-break-inside: avoid;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    overflow: hidden;
}

th {
    background: #0f172a;
    color: #ffffff;
    font-weight: 600;
    padding: 7px 10px;
    text-align: left;
    border: 1px solid #1e293b;
}

td {
    padding: 6px 10px;
    border: 1px solid #e2e8f0;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

/* ═══════════════════════════════════════════════
   CODE & MATH
   ═══════════════════════════════════════════════ */
pre {
    background: #090d16;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.2pt;
    padding: 12px 14px;
    border-radius: 6px;
    border: 1px solid #1e293b;
    overflow-x: auto;
    line-height: 1.45;
    margin: 12px 0;
    page-break-inside: avoid;
}

code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.8pt;
    background: #f1f5f9;
    color: #0f172a;
    padding: 1px 4px;
    border-radius: 3px;
    border: 1px solid #e2e8f0;
}

pre code {
    background: transparent;
    color: inherit;
    border: none;
    padding: 0;
}

.math-display {
    text-align: center;
    margin: 10px 0;
    padding: 8px 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.2pt;
    font-weight: 500;
    color: #0f172a;
    page-break-inside: avoid;
}

.formula-highlight {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #0369a1;
}

/* ═══════════════════════════════════════════════
   LISTS & ACCENTS
   ═══════════════════════════════════════════════ */
ul, ol {
    margin: 6px 0 12px 20px;
    padding: 0;
}

li {
    margin-bottom: 4px;
}

hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 22px 0;
}

.page-break {
    page-break-before: always;
}

.footer-note {
    font-size: 7.8pt;
    color: #94a3b8;
    text-align: center;
    margin-top: 30px;
    border-top: 1px solid #e2e8f0;
    padding-top: 10px;
}
"""

def clean_math(text):
    """Completely strips raw LaTeX tags and replaces them with clean human-readable text."""
    # Remove dollar signs
    text = text.replace("$", "")
    
    # Strip \text{...}
    text = re.sub(r"\\text\{([^\}]+)\}", r"\1", text)
    
    # Fractions: \frac{a}{b} -> (a / b)
    text = re.sub(r"\\frac\{([^\}]+)\}\{([^\}]+)\}", r"(\1 / \2)", text)
    
    # Square root: \sqrt{a} -> sqrt(a)
    text = re.sub(r"\\sqrt\{([^\}]+)\}", r"sqrt(\1)", text)
    
    # Subscripts and superscripts inside LaTeX: X_{abc} -> X_abc
    text = re.sub(r"_\{([^\}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^\}]+)\}", r"^\1", text)
    
    # Strip formatting tags: \, \; \! \quad \qquad \left \right
    text = re.sub(r"\\[,;!]", " ", text)
    text = re.sub(r"\\q?quad", " ", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = text.replace(r"\%", "%")
    
    # Greek and mathematical symbols to readable forms
    replacements = [
        (r"\\sigma_y", "Sigma_y"),
        (r"\\sigma_0", "Sigma_0"),
        (r"\\tau_{ss}", "Tau_ss"),
        (r"\\tau", "Tau"),
        (r"\\sigma", "Sigma"),
        (r"\\Delta G_mix", "Delta-G_mix"),
        (r"\\Delta H_mix", "Delta-H_mix"),
        (r"\\Delta S_mix", "Delta-S_mix"),
        (r"\\Delta S_config", "Delta-S_config"),
        (r"\\Delta E_f", "Delta-E_f"),
        (r"\\Delta", "Delta-"),
        (r"\\Omega", "Omega"),
        (r"\\delta", "delta"),
        (r"\\alpha_k\^\{?AB\}?", "alpha_k"),
        (r"\\alpha", "alpha"),
        (r"\\beta", "beta"),
        (r"\\gamma", "gamma"),
        (r"\\zeta_c", "zeta_c"),
        (r"\\Gamma", "Gamma"),
        (r"\\approx", "≈"),
        (r"\\le", "<="),
        (r"\\ge", ">="),
        (r"\\ll", "<<"),
        (r"\\gg", ">>"),
        (r"\\times", "*"),
        (r"\\cdot", "*"),
        (r"\\sum", "Sum"),
        (r"\\AA", "Å"),
        (r"\\sim", "~"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
        
    # Clean up double spaces
    text = re.sub(r" {2,}", " ", text)
    return text

def markdown_to_html(md_text, title="Document", badge="Academic Document"):
    lines = md_text.split('\n')
    html_lines = []
    
    in_code = False
    code_buffer = []
    in_table = False
    table_rows = []
    in_list = False
    list_type = 'ul'
    in_blockquote = False
    quote_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Code block toggle
        if stripped.startswith('```'):
            if not in_code:
                in_code = True
                code_buffer = []
            else:
                in_code = False
                escaped = html.escape('\n'.join(code_buffer))
                html_lines.append(f'<pre><code>{escaped}</code></pre>')
            continue
            
        if in_code:
            code_buffer.append(line)
            continue
            
        # Table rows
        if stripped.startswith('|') and stripped.endswith('|'):
            # Table separator check
            if re.match(r'^\|(\s*:?-+:?\s*\|)+$', stripped):
                continue
            cols = [c.strip() for c in stripped[1:-1].split('|')]
            table_rows.append(cols)
            in_table = True
            continue
        elif in_table:
            # Flush table
            if table_rows:
                header = table_rows[0]
                rows = table_rows[1:]
                th_html = ''.join(f'<th>{clean_math(c)}</th>' for c in header)
                tr_html = []
                for row in rows:
                    tds = ''.join(f'<td>{clean_math(c)}</td>' for c in row)
                    tr_html.append(f'<tr>{tds}</tr>')
                html_lines.append(f'<table><thead><tr>{th_html}</tr></thead><tbody>{"".join(tr_html)}</tbody></table>')
            table_rows = []
            in_table = False
            
        # Blockquote
        if stripped.startswith('>'):
            in_blockquote = True
            quote_lines.append(stripped[1:].strip())
            continue
        elif in_blockquote and not stripped.startswith('>'):
            if quote_lines:
                q_text = ' '.join(quote_lines)
                html_lines.append(f'<blockquote><p>{clean_math(q_text)}</p></blockquote>')
            quote_lines = []
            in_blockquote = False
            
        # Math blocks ($$ ... $$)
        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
            math_content = stripped[2:-2].strip()
            math_content = clean_math(math_content)
            html_lines.append(f'<div class="math-display">{math_content}</div>')
            continue
            
        # Headers
        if stripped.startswith('# '):
            html_lines.append(f'<h1>{clean_math(stripped[2:])}</h1>')
            continue
        elif stripped.startswith('## '):
            html_lines.append(f'<h2>{clean_math(stripped[3:])}</h2>')
            continue
        elif stripped.startswith('### '):
            html_lines.append(f'<h3>{clean_math(stripped[4:])}</h3>')
            continue
        elif stripped.startswith('#### '):
            html_lines.append(f'<h4>{clean_math(stripped[5:])}</h4>')
            continue
            
        # Horizontal rule
        if stripped in ('---', '***', '___'):
            html_lines.append('<hr>')
            continue
            
        # Unordered list
        if stripped.startswith('* ') or stripped.startswith('- '):
            item = stripped[2:].strip()
            # Inline bold
            item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
            item = clean_math(item)
            html_lines.append(f'<ul><li>{item}</li></ul>')
            continue
            
        # Ordered list
        m_num = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if m_num:
            item = m_num.group(2)
            item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
            item = clean_math(item)
            html_lines.append(f'<ol start="{m_num.group(1)}"><li>{item}</li></ol>')
            continue
            
        # Empty line
        if not stripped:
            continue
            
        # Paragraph
        p_text = stripped
        # Inline bold
        p_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', p_text)
        # Inline math $...$
        p_text = re.sub(r'\$([^$]+)\$', lambda m: f'<span class="formula-highlight">{clean_math(m.group(1))}</span>', p_text)
        p_text = clean_math(p_text)
        html_lines.append(f'<p>{p_text}</p>')

    # Final table flush if needed
    if in_table and table_rows:
        header = table_rows[0]
        rows = table_rows[1:]
        th_html = ''.join(f'<th>{clean_math(c)}</th>' for c in header)
        tr_html = []
        for row in rows:
            tds = ''.join(f'<td>{clean_math(c)}</td>' for c in row)
            tr_html.append(f'<tr>{tds}</tr>')
        html_lines.append(f'<table><thead><tr>{th_html}</tr></thead><tbody>{"".join(tr_html)}</tbody></table>')

    body_html = '\n'.join(html_lines)
    
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{html.escape(title)}</title>
    <style>
        {CSS_STYLES}
    </style>
</head>
<body>
    <div class="doc-header">
        <div class="doc-badge">{html.escape(badge)}</div>
        <h1 class="doc-title">{html.escape(title)}</h1>
        <div class="meta-grid">
            <div class="meta-item"><strong>Author:</strong> Solomon Ahedor</div>
            <div class="meta-item"><strong>Institution:</strong> KNUST (Materials & Metallurgical Eng.)</div>
            <div class="meta-item"><strong>Platform:</strong> MetaForge Discovery Engine</div>
            <div class="meta-item"><strong>Date:</strong> May 2026 (Revised Edition)</div>
        </div>
    </div>
    
    {body_html}
    
    <div class="footer-note">
        MetaForge Computational Framework · Solomon Ahedor · Department of Materials & Metallurgical Engineering, KNUST
    </div>
</body>
</html>
"""
    return full_html


def compile_pdf(html_path, output_pdf_path):
    """Invokes headless Microsoft Edge to render HTML to PDF."""
    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(output_pdf_path)
    
    cmd = [
        EDGE_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html.replace(os.sep, '/')}"
    ]
    
    print(f"[*] Rendering: {os.path.basename(output_pdf_path)}...")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if os.path.exists(abs_pdf) and os.path.getsize(abs_pdf) > 1000:
        size_kb = os.path.getsize(abs_pdf) / 1024
        print(f"    [+] SUCCESS: {os.path.basename(abs_pdf)} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"    [!] Error compiling {abs_pdf}: {proc.stderr}")
        return False


def main():
    print("=== MetaForge Academic PDF Generator ===")
    
    # 1. Proposal PDF
    prop_md_path = os.path.join(BASE_DIR, "RESEARCH_PROPOSAL.md")
    if os.path.exists(prop_md_path):
        with open(prop_md_path, 'r', encoding='utf-8') as f:
            prop_text = f.read()
        prop_html = markdown_to_html(
            prop_text, 
            title="Hierarchical Multi-Fidelity Inverse Design & Atomistic Validation of Multi-Principal Element Alloys",
            badge="Official Research Proposal · ICME Framework"
        )
        prop_html_path = os.path.join(BASE_DIR, "RESEARCH_PROPOSAL_RENDER.html")
        with open(prop_html_path, 'w', encoding='utf-8') as f:
            f.write(prop_html)
            
        prop_pdf_path = os.path.join(BASE_DIR, "Solomon_Ahedor_Research_Proposal_MetaForge.pdf")
        compile_pdf(prop_html_path, prop_pdf_path)

    # 2. Master Foundations Course PDF
    course_md_path = os.path.join(BASE_DIR, "COMPREHENSIVE_HEA_COURSE.md")
    if os.path.exists(course_md_path):
        with open(course_md_path, 'r', encoding='utf-8') as f:
            course_text = f.read()
        course_html = markdown_to_html(
            course_text,
            title="High-Entropy Alloys & Computational Materials Informatics: Master Foundations Course",
            badge="Doctoral Qualifying & Master Study Handbook"
        )
        course_html_path = os.path.join(BASE_DIR, "COMPREHENSIVE_COURSE_RENDER.html")
        with open(course_html_path, 'w', encoding='utf-8') as f:
            f.write(course_html)
            
        course_pdf_path = os.path.join(BASE_DIR, "Solomon_Ahedor_HEA_Master_Foundations_Course.pdf")
        compile_pdf(course_html_path, course_pdf_path)

    # 3. Full Publication Research Paper PDF
    paper_md_path = os.path.join(BASE_DIR, "PAPER_METAFORGE_DISCOVERY.md")
    if os.path.exists(paper_md_path):
        with open(paper_md_path, 'r', encoding='utf-8') as f:
            paper_text = f.read()
        paper_html = markdown_to_html(
            paper_text,
            title="Multi-Fidelity Inverse Design and Atomistic Validation of Multi-Principal Element Alloys for Extreme Engineering Environments",
            badge="Original Research Article · Materials Informatics"
        )
        paper_html_path = os.path.join(BASE_DIR, "RESEARCH_PAPER_RENDER.html")
        with open(paper_html_path, 'w', encoding='utf-8') as f:
            f.write(paper_html)
            
        paper_pdf_path = os.path.join(BASE_DIR, "Solomon_Ahedor_MetaForge_Research_Paper.pdf")
        compile_pdf(paper_html_path, paper_pdf_path)

    print("=== PDF Compilation Complete ===")

if __name__ == "__main__":
    main()

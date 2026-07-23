from __future__ import annotations

import base64
import html
import re
from io import BytesIO
from pathlib import Path

import markdown
import pykatex
import requests
from lxml import etree
from lxml import html as lxml_html
from PIL import Image
from premailer import transform
from pygments.formatters import HtmlFormatter


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "README.md"
HERO_IMAGE = HERE / "ppo.png"
OUTPUT = HERE / "ppo-wechat.html"

KATEX_VERSION = "0.16.22"
KATEX_CSS_URL = (
    f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist/katex.min.css"
)
KATEX_FONT_BASE = (
    f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist/fonts/"
)


# The source article uses "\\" as visual line separators. These equivalents
# retain the mathematics while using proper KaTeX alignment and notation.
DISPLAY_FORMULAS = [
    r"""
R_\theta
= \sum_\tau \operatorname{reward}(\tau)\,p_\theta(\tau)
""",
    r"""
\begin{aligned}
\nabla R_\theta
&= \sum_\tau \operatorname{reward}(\tau)\,\nabla p_\theta(\tau) \\
&= \sum_\tau \operatorname{reward}(\tau)\,p_\theta(\tau)
   \frac{\nabla p_\theta(\tau)}{p_\theta(\tau)} \\
&= \sum_\tau \operatorname{reward}(\tau)\,p_\theta(\tau)
   \nabla \log p_\theta(\tau) \\
&= \mathbb{E}_{\tau\sim p_\theta(\tau)}
   \left[\operatorname{reward}(\tau)\nabla\log p_\theta(\tau)\right] \\
&\approx \frac{1}{n}\sum_{\tau=1}^{n}
   \operatorname{reward}(\tau)\nabla\log p_\theta(\tau)
\end{aligned}
""",
    r"""
\begin{aligned}
p_\theta(\tau)
={}&p(s_{\tau 1})\,
p_\theta(a_{\tau 1}\mid s_{\tau 1})\,
p(s_{\tau 2}\mid s_{\tau 1},a_{\tau 1})\\
&{}\cdot p_\theta(a_{\tau 2}\mid s_{\tau 2})\cdots
\end{aligned}
""",
    r"""
\begin{aligned}
\nabla R_\theta
&= \frac{1}{n}\sum_{\tau=1}^{n}\operatorname{reward}(\tau)
   \sum_{t=1}^{T_\tau}
   \nabla\log p_\theta(a_{\tau t}\mid s_{\tau t}) \\
&= \frac{1}{n}\sum_{\tau=1}^{n}\sum_{t=1}^{T_\tau}
   \operatorname{reward}(\tau)
   \nabla\log p_\theta(a_{\tau t}\mid s_{\tau t})
\end{aligned}
""",
    r"""
\begin{aligned}
\nabla R_\theta
&= \sum_\tau \operatorname{reward}(\tau)\,p_\theta(\tau)
   \nabla\log p_\theta(\tau) \\
&= \sum_\tau \operatorname{reward}(\tau)\,p_{\theta'}(\tau)
   \frac{p_\theta(\tau)}{p_{\theta'}(\tau)}
   \nabla\log p_\theta(\tau) \\
&= \mathbb{E}_{\tau\sim p_{\theta'}(\tau)}
   \left[
   \frac{p_\theta(\tau)}{p_{\theta'}(\tau)}
   \operatorname{reward}(\tau)\nabla\log p_\theta(\tau)
   \right] \\
&\approx \frac{1}{n}\sum_{\tau=1}^{n}
   \frac{p_\theta(\tau)}{p_{\theta'}(\tau)}
   \operatorname{reward}(\tau)\nabla\log p_\theta(\tau),
   \qquad \text{使用 }\theta'\text{ 采样} \\
&= \frac{1}{n}\sum_{\tau=1}^{n}
   \frac{
     p(s_{\tau1})p_\theta(a_{\tau1}\mid s_{\tau1})
     p(s_{\tau2}\mid s_{\tau1},a_{\tau1})
     p_\theta(a_{\tau2}\mid s_{\tau2})\cdots
   }{
     p(s_{\tau1})p_{\theta'}(a_{\tau1}\mid s_{\tau1})
     p(s_{\tau2}\mid s_{\tau1},a_{\tau1})
     p_{\theta'}(a_{\tau2}\mid s_{\tau2})\cdots
   } \\
&\qquad\cdot \operatorname{reward}(\tau)
   \sum_{t=1}^{T_\tau}\nabla\log p_\theta(a_{\tau t}\mid s_{\tau t}) \\
&\approx \frac{1}{n}\sum_{\tau=1}^{n}\sum_{t=1}^{T_\tau}
   \frac{p_\theta(a_{\tau t}\mid s_{\tau t})}
        {p_{\theta'}(a_{\tau t}\mid s_{\tau t})}
   \operatorname{reward}(\tau)
   \nabla\log p_\theta(a_{\tau t}\mid s_{\tau t})
\end{aligned}
""",
    r"""
\begin{aligned}
\nabla J_{\mathrm{PPO}}^{\theta'}(\theta)
={}&\frac{1}{n}\sum_{\tau=1}^{n}\sum_{t=1}^{T_\tau}
\frac{p_\theta(a_{\tau t}\mid s_{\tau t})}
     {p_{\theta'}(a_{\tau t}\mid s_{\tau t})}
\operatorname{reward}(\tau)
\nabla\log p_\theta(a_{\tau t}\mid s_{\tau t}) \\
&{}-\beta\nabla D_{\mathrm{KL}}(\theta,\theta')
\end{aligned}
""",
    r"""
\begin{aligned}
J_{\mathrm{PPO}}^{\theta'}
= \sum_{(s,a)}
\min\left(
\begin{array}{l}
\dfrac{p_\theta(a\mid s)}{p_{\theta'}(a\mid s)}
A^{\theta'}(s,a), \\[6pt]
\operatorname{clip}\!\left(
\dfrac{p_\theta(a\mid s)}{p_{\theta'}(a\mid s)},
1-\epsilon,1+\epsilon
\right)A^{\theta'}(s,a)
\end{array}
\right)
-\beta D_{\mathrm{KL}}(\theta,\theta')
\end{aligned}
""",
    r"""
\begin{aligned}
J_{\mathrm{GRPO}}^{\theta'}
= \frac{1}{G}\sum_{i=1}^{G}
\min\left(
\begin{array}{l}
\dfrac{p_\theta(o_i\mid q)}{p_{\theta'}(o_i\mid q)}
A^{\theta'}(q,o_i), \\[6pt]
\operatorname{clip}\!\left(
\dfrac{p_\theta(o_i\mid q)}{p_{\theta'}(o_i\mid q)},
1-\epsilon,1+\epsilon
\right)A^{\theta'}(q,o_i)
\end{array}
\right)
-\beta D_{\mathrm{KL}}(\theta,\theta')
\end{aligned}
""",
    r"""
A_i
= \frac{
r_i-\operatorname{mean}\!\left(\{r_1,r_2,\ldots,r_G\}\right)
}{
\operatorname{std}\!\left(\{r_1,r_2,\ldots,r_G\}\right)
}
""",
]


ARTICLE_CSS = r"""
.wx-article {
  box-sizing: border-box;
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  padding: 48px 48px 56px;
  background: #ffffff;
  color: #303447;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  font-size: 16px;
  line-height: 1.9;
  letter-spacing: 0.025em;
  word-break: break-word;
}
.wx-article * {
  box-sizing: border-box;
}
.wx-article .article-header {
  margin: 0 0 30px;
  padding: 0;
}
.wx-article .eyebrow {
  display: inline-block;
  margin: 0 0 14px;
  padding: 5px 11px;
  border: 1px solid #dcdafa;
  border-radius: 999px;
  background: #f5f4ff;
  color: #5e57bf;
  font-size: 11px;
  font-weight: 700;
  line-height: 1.4;
  letter-spacing: 0.14em;
}
.wx-article h1 {
  margin: 0;
  color: #1e2340;
  font-size: 30px;
  font-weight: 800;
  line-height: 1.35;
  letter-spacing: -0.02em;
}
.wx-article .title-rule {
  display: block;
  width: 44px;
  height: 4px;
  margin: 20px 0 0;
  border-radius: 4px;
  background: #7067d9;
}
.wx-article .hero {
  margin: 0 0 30px;
  padding: 0;
}
.wx-article .hero-image {
  display: block;
  width: 100%;
  height: auto;
  margin: 0;
  border: 0;
  border-radius: 16px;
}
.wx-article .hero-caption {
  margin: 10px 2px 0;
  color: #9296a8;
  font-size: 12px;
  line-height: 1.7;
  text-align: center;
  letter-spacing: 0.04em;
}
.wx-article .reading-route {
  margin: 0 0 30px;
  padding: 13px 16px;
  border: 1px solid #ececf5;
  border-radius: 10px;
  background: #fafafd;
  color: #686d83;
  font-size: 12px;
  line-height: 1.7;
  text-align: center;
  letter-spacing: 0.055em;
}
.wx-article .route-arrow {
  display: inline-block;
  margin: 0 8px;
  color: #aaa6db;
}
.wx-article p {
  margin: 0 0 22px;
  color: #363b4f;
  font-size: 16px;
  line-height: 1.95;
  text-align: justify;
  text-justify: inter-ideograph;
}
.wx-article .lead {
  margin: 0 0 34px;
  padding: 19px 20px;
  border-left: 4px solid #7067d9;
  border-radius: 0 10px 10px 0;
  background: #f7f7ff;
  color: #414760;
  font-size: 16px;
  line-height: 1.9;
}
.wx-article h3 {
  display: flex;
  align-items: center;
  margin: 50px 0 24px;
  padding: 0 0 12px;
  border-bottom: 1px solid #e9e9f2;
  color: #202541;
  font-size: 22px;
  font-weight: 750;
  line-height: 1.45;
  letter-spacing: -0.01em;
}
.wx-article .section-index {
  display: inline-block;
  flex: 0 0 auto;
  margin: 0 12px 0 0;
  padding: 4px 8px;
  border-radius: 6px;
  background: #282d55;
  color: #ffffff;
  font-size: 12px;
  font-weight: 800;
  line-height: 1.45;
  letter-spacing: 0.08em;
}
.wx-article .section-title {
  display: inline-block;
}
.wx-article strong {
  color: #262b49;
  font-weight: 700;
}
.wx-article a {
  color: #5c55bc;
  text-decoration: none;
  border-bottom: 1px solid #bbb7ea;
}
.wx-article .math-inline {
  display: inline-block;
  max-width: 100%;
  margin: 0 0.08em;
  color: #413b9d;
  line-height: 1.25;
  vertical-align: -0.08em;
  white-space: nowrap;
}
.wx-article .math-block {
  width: 100%;
  margin: 28px 0 30px;
  padding: 15px 17px 17px;
  border: 1px solid #e4e2f8;
  border-left: 3px solid #7067d9;
  border-radius: 10px;
  background: #fbfaff;
  color: #24294a;
}
.wx-article .math-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0 0 10px;
  color: #8a86b8;
  font-size: 10px;
  font-weight: 700;
  line-height: 1.4;
  letter-spacing: 0.13em;
}
.wx-article .math-number {
  color: #aaa7c8;
}
.wx-article .math-scroll {
  display: block;
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 4px 1px 6px;
  -webkit-overflow-scrolling: touch;
}
.wx-article .math-scroll .katex-display {
  display: block;
  min-width: max-content;
  margin: 0;
  text-align: left;
}
.wx-article .demo-callout {
  margin: 30px 0;
  padding: 19px 20px;
  border: 1px solid #dfe8ee;
  border-radius: 10px;
  background: #f5fafb;
  color: #334a53;
}
.wx-article .code-card {
  width: 100%;
  margin: 25px 0 32px;
  overflow: hidden;
  border: 1px solid #262b45;
  border-radius: 11px;
  background: #171b2d;
}
.wx-article .code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0;
  padding: 10px 14px;
  border-bottom: 1px solid #303650;
  background: #20253c;
  color: #b8bdd0;
  font-size: 10px;
  font-weight: 700;
  line-height: 1.4;
  letter-spacing: 0.1em;
}
.wx-article .code-dots {
  color: #777f9c;
  letter-spacing: 0.2em;
}
.wx-article .codehilite {
  width: 100%;
  margin: 0;
  padding: 0;
  background: #171b2d;
}
.wx-article .codehilite pre {
  width: 100%;
  max-width: 100%;
  margin: 0;
  padding: 18px 18px 20px;
  overflow-x: auto;
  background: #171b2d;
  color: #d7dbeb;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 12px;
  line-height: 1.7;
  letter-spacing: 0;
  tab-size: 4;
  white-space: pre;
  -webkit-overflow-scrolling: touch;
}
.wx-article .codehilite code {
  margin: 0;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  white-space: pre;
}
.wx-article p code {
  margin: 0 2px;
  padding: 2px 5px;
  border-radius: 4px;
  background: #f0eff8;
  color: #514aaf;
  font-family: Consolas, "Liberation Mono", monospace;
  font-size: 0.88em;
}
.wx-article .article-footer {
  margin: 54px 0 0;
  padding: 23px 0 0;
  border-top: 1px solid #e9e9f2;
  color: #9a9daf;
  font-size: 11px;
  line-height: 1.8;
  text-align: center;
  letter-spacing: 0.18em;
}
"""


PREVIEW_CSS = r"""
html, body {
  margin: 0;
  min-height: 100%;
  background: #eef0f5;
}
body {
  padding: 88px 20px 48px;
}
.preview-toolbar {
  position: fixed;
  z-index: 9999;
  top: 0;
  right: 0;
  left: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 64px;
  padding: 10px 24px;
  background: rgba(30, 35, 64, 0.97);
  box-shadow: 0 5px 20px rgba(27, 31, 54, 0.2);
  color: #ffffff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei",
    sans-serif;
}
.toolbar-title {
  font-size: 14px;
  font-weight: 700;
}
.toolbar-meta {
  margin-top: 2px;
  color: #aeb3ce;
  font-size: 11px;
}
.copy-button {
  padding: 10px 16px;
  border: 0;
  border-radius: 8px;
  background: #766de0;
  color: #ffffff;
  cursor: pointer;
  font-size: 13px;
  font-weight: 700;
}
.copy-button:hover {
  background: #847be8;
}
.preview-canvas {
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  overflow: hidden;
  border-radius: 4px;
  box-shadow: 0 18px 55px rgba(37, 42, 69, 0.13);
  background: #ffffff;
}
.toast {
  position: fixed;
  z-index: 10000;
  right: 22px;
  bottom: 22px;
  padding: 11px 15px;
  border-radius: 8px;
  background: #242a49;
  box-shadow: 0 8px 24px rgba(22, 25, 44, 0.25);
  color: #ffffff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei",
    sans-serif;
  font-size: 13px;
  opacity: 0;
  pointer-events: none;
  transform: translateY(8px);
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.toast.show {
  opacity: 1;
  transform: translateY(0);
}
@media (max-width: 640px) {
  body {
    padding: 74px 0 0;
  }
  .preview-toolbar {
    min-height: 58px;
    padding: 8px 12px;
  }
  .toolbar-meta {
    display: none;
  }
  .preview-canvas {
    border-radius: 0;
    box-shadow: none;
  }
  .wx-article {
    padding: 34px 20px 46px !important;
  }
  .wx-article h1 {
    font-size: 27px !important;
  }
}
@media print {
  body {
    padding: 0;
    background: #ffffff;
  }
  .preview-toolbar, .toast {
    display: none !important;
  }
  .preview-canvas {
    max-width: none;
    box-shadow: none;
  }
}
"""


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>__PAGE_TITLE__</title>
  <style>
__KATEX_CSS__
__ARTICLE_CSS__
__PYGMENTS_CSS__
__PREVIEW_CSS__
  </style>
</head>
<body>
  <aside class="preview-toolbar" data-preview-only="true">
    <div>
      <div class="toolbar-title">微信公众号排版预览</div>
      <div class="toolbar-meta">__FORMULA_SUMMARY__ · 已静态渲染并内联样式</div>
    </div>
    <button class="copy-button" id="copy-article" type="button">复制公众号正文</button>
  </aside>
  <main class="preview-canvas">
__INLINED_ARTICLE__
  </main>
  <div class="toast" id="copy-toast" role="status">正文已复制，可粘贴到公众号编辑器</div>
  <script>
    (function () {
      const button = document.getElementById("copy-article");
      const article = document.getElementById("wechat-article");
      const toast = document.getElementById("copy-toast");

      function cleanClone() {
        const clone = article.cloneNode(true);
        clone.removeAttribute("id");
        clone.querySelectorAll("[data-tex]").forEach(function (node) {
          node.removeAttribute("data-tex");
          node.removeAttribute("title");
        });
        return clone;
      }

      function fallbackCopy(clone) {
        const holder = document.createElement("div");
        holder.setAttribute("contenteditable", "true");
        holder.style.position = "fixed";
        holder.style.left = "-100000px";
        holder.style.top = "0";
        holder.appendChild(clone);
        document.body.appendChild(holder);
        const range = document.createRange();
        range.selectNodeContents(holder);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        const copied = document.execCommand("copy");
        selection.removeAllRanges();
        holder.remove();
        if (!copied) {
          throw new Error("copy command failed");
        }
      }

      async function copyArticle() {
        const clone = cleanClone();
        const richHtml = '<meta charset="utf-8">' + clone.outerHTML;
        const plainText = article.innerText;

        try {
          if (navigator.clipboard && window.ClipboardItem) {
            await navigator.clipboard.write([
              new ClipboardItem({
                "text/html": new Blob([richHtml], { type: "text/html" }),
                "text/plain": new Blob([plainText], { type: "text/plain" })
              })
            ]);
          } else {
            fallbackCopy(clone);
          }
          toast.textContent = "正文已复制，可粘贴到公众号编辑器";
        } catch (error) {
          try {
            fallbackCopy(clone);
            toast.textContent = "正文已复制，可粘贴到公众号编辑器";
          } catch (fallbackError) {
            toast.textContent = "复制失败，请手动选中正文复制";
          }
        }

        toast.classList.add("show");
        window.setTimeout(function () {
          toast.classList.remove("show");
        }, 2200);
      }

      button.addEventListener("click", copyArticle);
    }());
  </script>
</body>
</html>
"""


def normalize_inline_formula(tex: str) -> str:
    tex = tex.strip()
    exact_replacements = {
        r"\{o1, o2, ..., o_G\}": r"\{o_1,o_2,\ldots,o_G\}",
        r"{o1, o2, ..., o_G}": r"\{o_1,o_2,\ldots,o_G\}",
    }
    tex = exact_replacements.get(tex, tex)
    tex = re.sub(r"(?<!\\)logp", r"\\log p", tex)
    tex = re.sub(r"(?<!\\)\|", r"\\mid ", tex)
    tex = tex.replace("...", r"\ldots")
    return tex


def extract_math(source: str) -> tuple[str, list[dict[str, object]]]:
    snippets: list[dict[str, object]] = []
    display_index = 0

    def replace_display(match: re.Match[str]) -> str:
        nonlocal display_index
        if display_index >= len(DISPLAY_FORMULAS):
            raise ValueError("Source contains more display formulas than expected")
        token = f"MATHDISPLAYTOKEN{display_index:03d}END"
        snippets.append(
            {
                "token": token,
                "tex": DISPLAY_FORMULAS[display_index].strip(),
                "display": True,
                "number": display_index + 1,
            }
        )
        display_index += 1
        return token

    prepared = re.sub(r"(?<!\\)\$\$(.*?)(?<!\\)\$\$", replace_display, source, flags=re.S)

    inline_index = 0

    def replace_inline(match: re.Match[str]) -> str:
        nonlocal inline_index
        token = f"MATHINLINETOKEN{inline_index:03d}END"
        snippets.append(
            {
                "token": token,
                "tex": normalize_inline_formula(match.group(1)),
                "display": False,
                "number": inline_index + 1,
            }
        )
        inline_index += 1
        return token

    prepared = re.sub(
        r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$",
        replace_inline,
        prepared,
        flags=re.S,
    )

    if display_index != len(DISPLAY_FORMULAS):
        raise ValueError(
            f"Expected {len(DISPLAY_FORMULAS)} display formulas, found {display_index}"
        )
    return prepared, snippets


def decorate_markdown(rendered: str) -> str:
    root = lxml_html.fragment_fromstring(rendered, create_parent="div")

    paragraphs = root.xpath("./p")
    if paragraphs:
        paragraphs[0].set("class", "lead")

    for index, heading in enumerate(root.cssselect("h3"), start=1):
        heading_text = "".join(heading.itertext()).strip()
        heading.clear()
        number = etree.SubElement(heading, "span", {"class": "section-index"})
        number.text = f"{index:02d}"
        label = etree.SubElement(heading, "span", {"class": "section-title"})
        label.text = re.sub(r"^\d+\.\s*", "", heading_text)

    for paragraph in root.cssselect("p"):
        if "policy-gradient-demo" in "".join(paragraph.itertext()):
            current_class = paragraph.get("class", "")
            paragraph.set("class", f"{current_class} demo-callout".strip())

    code_labels = [
        ("PYTHON", "PPO 训练主循环"),
        ("PROMPT", "思维链模板"),
        ("OUTPUT", "示例输出"),
    ]
    for block, (kind, label) in zip(root.cssselect("div.codehilite"), code_labels):
        parent = block.getparent()
        position = parent.index(block)
        parent.remove(block)

        wrapper = etree.Element("section", {"class": "code-card"})
        header = etree.SubElement(wrapper, "div", {"class": "code-header"})
        name = etree.SubElement(header, "span")
        name.text = f"{kind} · {label}"
        dots = etree.SubElement(header, "span", {"class": "code-dots"})
        dots.text = "● ● ●"
        wrapper.append(block)
        parent.insert(position, wrapper)

    return "".join(
        etree.tostring(child, encoding="unicode", method="html") for child in root
    )


def render_formula(tex: str, display: bool, number: int) -> str:
    rendered = pykatex.renderToString(
        tex,
        displayMode=display,
        output=pykatex.OUTPUT_HTML,
        throwOnError=True,
        strict=pykatex.STRICT_WARN,
    )
    escaped_tex = html.escape(tex, quote=True)

    if display:
        return (
            f'<section class="math-block" data-tex="{escaped_tex}">'
            '<div class="math-meta"><span>FORMULA</span>'
            f'<span class="math-number">{number:02d}</span></div>'
            f'<div class="math-scroll">{rendered}</div>'
            "</section>"
        )
    return (
        f'<span class="math-inline" data-tex="{escaped_tex}" '
        f'title="{escaped_tex}">{rendered}</span>'
    )


def inject_math(rendered: str, snippets: list[dict[str, object]]) -> str:
    result = rendered
    for snippet in snippets:
        token = str(snippet["token"])
        display = bool(snippet["display"])
        formula_html = render_formula(
            str(snippet["tex"]), display, int(snippet["number"])
        )
        if display:
            result = result.replace(f"<p>{token}</p>", formula_html)
        result = result.replace(token, formula_html)

    leftovers = re.findall(r"MATH(?:DISPLAY|INLINE)TOKEN\d{3}END", result)
    if leftovers:
        raise ValueError(f"Unreplaced math placeholders: {leftovers}")
    return result


def get_katex_css() -> str:
    response = requests.get(KATEX_CSS_URL, timeout=30)
    response.raise_for_status()
    return response.text.replace("url(fonts/", f"url({KATEX_FONT_BASE}")


def image_data_uri(path: Path) -> str:
    buffer = BytesIO()
    with Image.open(path) as source:
        source.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=88,
            optimize=True,
            progressive=True,
        )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_article(title: str, body_html: str) -> str:
    hero_uri = image_data_uri(HERO_IMAGE)
    display_title = "从零写一个 PPO：强化学习训练吃豆人"
    return f"""
<article id="wechat-article" class="wx-article">
  <header class="article-header">
    <span class="eyebrow">REINFORCEMENT LEARNING · HANDS-ON</span>
    <h1>{html.escape(display_title)}</h1>
    <span class="title-rule"></span>
  </header>
  <figure class="hero">
    <img class="hero-image" src="{hero_uri}" alt="PPO 强化学习文章封面">
    <figcaption class="hero-caption">从策略梯度出发，逐层搭起 PPO 与 GRPO 的推导阶梯</figcaption>
  </figure>
  <section class="reading-route">
    <span>Policy Gradient</span><span class="route-arrow">→</span>
    <span>PPO</span><span class="route-arrow">→</span>
    <span>GRPO</span><span class="route-arrow">→</span>
    <span>Reward</span>
  </section>
  {body_html}
  <footer class="article-footer">THE END · KEEP SCALING</footer>
</article>
""".strip()


def inline_article(article: str, katex_css: str, pygments_css: str) -> str:
    # @font-face cannot survive a WeChat paste, while all structural KaTeX
    # styles can. The full font declarations remain in the preview document.
    katex_without_fonts = re.sub(
        r"@font-face\s*\{[^{}]*\}", "", katex_css, flags=re.I
    )
    inlined_document = transform(
        article,
        css_text=f"{katex_without_fonts}\n{ARTICLE_CSS}\n{pygments_css}",
        remove_classes=False,
        keep_style_tags=False,
        strip_important=False,
        disable_validation=True,
    )
    parsed = lxml_html.fromstring(inlined_document)
    matched = parsed.xpath('//*[@id="wechat-article"]')
    if not matched:
        raise ValueError("Could not find the inlined article")
    return etree.tostring(matched[0], encoding="unicode", method="html")


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    source_title = re.sub(r"^#+\s*", "", source_lines[0]).strip()
    markdown_body = "\n".join(source_lines[1:]).strip()
    markdown_body = markdown_body.replace(
        "链接：https://github.com/xytpai/policy-gradient-demo。",
        "链接：[policy-gradient-demo](https://github.com/xytpai/policy-gradient-demo)。",
    )

    prepared, snippets = extract_math(markdown_body)
    rendered_markdown = markdown.markdown(
        prepared,
        extensions=["extra", "codehilite", "toc"],
        extension_configs={
            "codehilite": {
                "css_class": "codehilite",
                "guess_lang": False,
                "linenums": False,
            }
        },
        output_format="html5",
    )
    decorated = decorate_markdown(rendered_markdown)
    with_math = inject_math(decorated, snippets)

    katex_css = get_katex_css()
    pygments_css = HtmlFormatter(style="material").get_style_defs(".codehilite")
    article = build_article(source_title, with_math)
    inlined_article = inline_article(article, katex_css, pygments_css)

    display_count = sum(bool(item["display"]) for item in snippets)
    inline_count = len(snippets) - display_count
    output = (
        PAGE_TEMPLATE.replace("__PAGE_TITLE__", html.escape(source_title))
        .replace("__KATEX_CSS__", katex_css)
        .replace("__ARTICLE_CSS__", ARTICLE_CSS)
        .replace("__PYGMENTS_CSS__", pygments_css)
        .replace("__PREVIEW_CSS__", PREVIEW_CSS)
        .replace("__FORMULA_SUMMARY__", f"{display_count} 个大公式 · {inline_count} 个行内公式")
        .replace("__INLINED_ARTICLE__", inlined_article)
    )

    OUTPUT.write_text(output, encoding="utf-8")
    print(f"Rendered {OUTPUT}")
    print(f"Static formulas: {display_count} display + {inline_count} inline")


if __name__ == "__main__":
    main()

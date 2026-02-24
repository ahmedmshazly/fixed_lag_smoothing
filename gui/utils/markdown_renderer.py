"""
gui/utils/markdown_renderer.py
Standalone utility for rendering simple Markdown into a Tkinter Text widget.
"""
import re
import tkinter as tk
from typing import Optional


class MarkdownRenderer:
    """Handles the parsing and insertion of markdown text into Tkinter Text widgets."""

    @staticmethod
    def strip_latex_delimiters(text: str) -> str:
        """Removes LaTeX inline ($...$) and display ($$...$$) delimiters."""
        text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\$(.*?)\$', r'\1', text, flags=re.DOTALL)
        return text

    @classmethod
    def render(cls, text: str, widget: tk.Text) -> None:
        """Parses markdown and inserts it into the target widget with appropriate tags."""
        text = cls.strip_latex_delimiters(text)
        is_disabled = widget.cget("state") == tk.DISABLED

        if is_disabled:
            widget.configure(state=tk.NORMAL)

        for line in text.splitlines():
            line = line.rstrip()
            if not line:
                widget.insert(tk.END, "\n")
                continue

            cls._parse_line(line, widget)

        if is_disabled:
            widget.configure(state=tk.DISABLED)
        widget.see(tk.END)

    @classmethod
    def _parse_line(cls, line: str, widget: tk.Text) -> None:
        """Parses a single line of markdown and applies structural tags."""
        if line.startswith("### "):
            widget.insert(tk.END, line[4:] + "\n", "heading3")
        elif line.startswith("## "):
            widget.insert(tk.END, line[3:] + "\n", "heading2")
        elif line.startswith("# "):
            widget.insert(tk.END, line[2:] + "\n", "heading1")
        elif line.startswith("- ") or line.startswith("* ") or line.startswith("● "):
            widget.insert(tk.END, "  • " + line[2:] + "\n", "bullet")
        elif re.match(r'^\d+\.\s', line):
            widget.insert(tk.END, "  " + line + "\n", "bullet")
        elif line.startswith("---"):
            widget.insert(tk.END, "─" * 50 + "\n", "hr")
        elif line.startswith("    "):
            widget.insert(tk.END, line + "\n", "code")
        else:
            cls._parse_inline_formatting(line, widget)
            widget.insert(tk.END, "\n")

    @classmethod
    def _parse_inline_formatting(cls, line: str, widget: tk.Text) -> None:
        """Handles **bold** and *italic* formatting within a line."""
        # Simplified for brevity; robust parsers might use a state machine or regex.
        parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', line)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                widget.insert(tk.END, part[2:-2], "bold")
            elif part.startswith("*") and part.endswith("*") and not part == "*":
                widget.insert(tk.END, part[1:-1], "italic")
            else:
                widget.insert(tk.END, part)

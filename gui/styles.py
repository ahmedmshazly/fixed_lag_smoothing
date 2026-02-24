"""
gui/styles.py
Centralized UI styling and configuration.
"""
import tkinter as tk


class UITheme:
    """Holds color palettes and font configurations for the application."""
    COLORS = {
        "bg_light": "#f8f9fa",
        "primary": "#1a5276",
        "secondary": "#2874a6",
        "accent": "#3498db",
        "insight_fg": "#0052cc",
        "code_bg": "#ecf0f1",
        "hr": "#7f8c8d"
    }

    FONTS = {
        "base": ("Arial", 10),
        "italic": ("Arial", 10, "italic"),
        "mono": ("Consolas", 10),
        "mono_bold": ("Consolas", 10, "bold"),
        "h1": ("Consolas", 14, "bold"),
        "h2": ("Consolas", 12, "bold"),
        "h3": ("Consolas", 11, "bold"),
    }

    @classmethod
    def apply_markdown_tags(cls, text_widget: tk.Text) -> None:
        """Configures a Tkinter Text widget with markdown-style text tags."""
        text_widget.tag_configure("heading1", font=cls.FONTS["h1"], foreground=cls.COLORS["primary"])
        text_widget.tag_configure("heading2", font=cls.FONTS["h2"], foreground=cls.COLORS["secondary"])
        text_widget.tag_configure("heading3", font=cls.FONTS["h3"], foreground=cls.COLORS["accent"])
        text_widget.tag_configure("bold", font=cls.FONTS["mono_bold"])
        text_widget.tag_configure("italic", font=cls.FONTS["italic"])
        text_widget.tag_configure("code", font=cls.FONTS["mono"], background=cls.COLORS["code_bg"])
        text_widget.tag_configure("bullet", lmargin1=20, lmargin2=30)
        text_widget.tag_configure("hr", foreground=cls.COLORS["hr"], font=cls.FONTS["mono"])

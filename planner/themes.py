"""The two app themes. .streamlit/config.toml must match these (tests/test_themes.py checks it).

light = Indigo (indigo accents, green success colours)
dark  = Sunset (warm orange on zinc)
"""

FONT = "Plus Jakarta Sans:https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap"

THEMES = {
    "light": {
        "label": "Indigo",
        "primaryColor": "#4F46E5",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F5F7FB",
        "textColor": "#111827",
        "borderColor": "#E5E7EB",
        "linkColor": "#4F46E5",
        "greenColor": "#16A34A",
        "excel_accent": "4F46E5",
    },
    "dark": {
        "label": "Sunset",
        "primaryColor": "#F97316",
        "backgroundColor": "#18181B",
        "secondaryBackgroundColor": "#27272A",
        "textColor": "#FAFAFA",
        "borderColor": "#3F3F46",
        "linkColor": "#FB923C",
        "greenColor": "#4ADE80",
        "excel_accent": "EA580C",
    },
}

CONFIG_KEYS = ("primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor", "borderColor",
               "linkColor", "greenColor")


def config_toml() -> str:
    lines = ["# Generated from planner/themes.py (tests/test_themes.py keeps them in sync)", "",
             "[theme]", f'font = "{FONT}"', 'baseRadius = "0.6rem"', 'buttonRadius = "0.6rem"',
             "showWidgetBorder = true", "showSidebarBorder = true", ""]
    for mode in ("light", "dark"):
        lines.append(f"[theme.{mode}]")
        lines += [f'{k} = "{THEMES[mode][k]}"' for k in CONFIG_KEYS]
        lines.append("")
    lines += ["[server]", "maxUploadSize = 20", "", "[browser]", "gatherUsageStats = false", ""]
    return "\n".join(lines)

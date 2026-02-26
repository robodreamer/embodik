"""MkDocs plugin that injects version from the installed package into config and markdown."""

from __future__ import annotations

import mkdocs.plugins

from embodik import __version__


class VersionPlugin(mkdocs.plugins.BasePlugin):
    """Inject package version into docs (single source of truth: pyproject.toml)."""

    def on_config(self, config, **kwargs):
        """Inject version into config.extra for theme/plugins."""
        config.extra["version"] = __version__
        return config

    def on_page_markdown(self, markdown, page, config, files):
        """Replace {{ version }} placeholder in markdown with actual version."""
        version = config.extra.get("version", "")
        return markdown.replace("{{ version }}", version)

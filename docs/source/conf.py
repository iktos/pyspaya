import sys
import sphinx_rtd_theme
import os
import re


def find_release_version() -> str:
    find_version = re.compile(r"^version *= *\"(.*)\"$")
    with open("../../pyproject.toml", "r") as toml:
        for line in toml.readlines():
            version_found = find_version.search(line)
            if version_found is not None:
                return version_found.group(1)
    return "0.1.0"


sys.path.append(os.path.abspath("../.."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
]


autodoc_default_options = {"members": True, "inherited-members": True}

source_suffix = ".rst"
master_doc = "index"

author = "IKTOS"
project = "Spaya API Client"

release = find_release_version()
version = f"Tag ({release})"

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_show_sourcelink = False

intersphinx_mapping = {"python": ("https://docs.python.org/", None)}
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
}

latex_domain_indices = False
latex_elements = {"releasename": "Version", "printindex": ""}

add_module_names = False

autoclass_content = "both"
autodoc_typehints = "description"

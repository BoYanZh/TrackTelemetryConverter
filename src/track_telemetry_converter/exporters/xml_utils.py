"""XML helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def indent_xml(tree, space="  ", level=0):
    """Indent an XML tree in place (xml.etree.ElementTree.indent)."""
    ET.indent(tree, space=space, level=level)

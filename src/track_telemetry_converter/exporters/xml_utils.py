"""XML helpers compatible with every supported Python version."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def indent_xml(tree, space="  ", level=0):
    """Indent an XML tree, including on Python 3.8 where ET.indent is absent."""
    stdlib_indent = getattr(ET, "indent", None)
    if stdlib_indent is not None:
        stdlib_indent(tree, space=space, level=level)
        return

    element = tree.getroot() if isinstance(tree, ET.ElementTree) else tree

    def indent_element(current, current_level):
        children = list(current)
        if not children:
            return

        child_level = current_level + 1
        if not current.text or not current.text.strip():
            current.text = "\n" + space * child_level

        for index, child in enumerate(children):
            indent_element(child, child_level)
            if not child.tail or not child.tail.strip():
                tail_level = child_level if index < len(children) - 1 else current_level
                child.tail = "\n" + space * tail_level

    indent_element(element, level)

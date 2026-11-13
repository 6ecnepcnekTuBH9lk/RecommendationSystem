"""Central Qt-free table help, keyed by stable semantic IDs, never display titles."""
from dataclasses import dataclass
from base64 import b64encode
from functools import lru_cache
from html import escape

from Application.paths import ICONS_DIR

EXPLANATION_ICON = ICONS_DIR / "question.png"


@dataclass(frozen=True)
class TableExplanation:
    description: str
    terms: tuple[tuple[str, str], ...]
    limitation: str = ""

    def html(self):
        # Text is escaped here for both Qt rich text and offline HTML.
        body = "<p>" + escape(self.description) + "</p><ul>"
        body += "".join("<li><b>" + escape(term) + ":</b> " + escape(text) + "</li>" for term, text in self.terms)
        return body + "</ul>" + ("<p>" + escape(self.limitation) + "</p>" if self.limitation else "")


# Assign only after a table's explanation has been agreed. No current UI table opts in.
TABLE_EXPLANATIONS: dict[str, TableExplanation] = {}


@lru_cache(maxsize=1)
def explanation_icon_data():
    return "data:image/png;base64," + b64encode(EXPLANATION_ICON.read_bytes()).decode("ascii")


def explanation_html(table_id, explanation):
    return ('<details class="table-explanation" data-table-id="' + escape(table_id, quote=True) + '">'
            '<summary><img src="' + explanation_icon_data() + '" width="20" height="20" alt="">'
            'Как читать таблицу</summary>' + explanation.html() + '</details>')


def table_explanation(table_id):
    return TABLE_EXPLANATIONS.get(table_id)

"""UI06: bounded history pages and recommendation display, without index builds."""

from threading import Event, get_ident
import subprocess
import sys

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QStyle, QStyleOptionViewItem

from Application.customer_card_data import ProductContent
from Application.customer_history_normalization import normalize_action
from Application.customer_history_presentation import HistoryRow, present_history
from Application.customer_history_records import HistoryKind as K
from Application.customer_history_service import HistoryPage, HistoryStatus as S, HistoryVariant, LogicalHistoryEvent
from Application.mindbox.identity import CustomerIdResolver
from Application.model.raw_bpr_inference import RawBprRecommendation
from Application.tabs.customer_card_widgets import ProductCard
from Application.tabs.customer_history_widgets import MAX_EMPTY_PAGES
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_customer_results_card import app as app, card as card, wait_for
from tests.test_customer_history_index import action


def page(kind, products, *, cursor=None, key="event", status=S.READY, customer="C", generation="g"):
    raw = action(key, customer=customer, products=products)
    if kind is K.FAVORITE:
        raw['actionTemplate']['ids']['systemName'] = "DobavlenieProduktaVSpisokVOperaciiDobavlenie"
    event = normalize_action(raw, CustomerIdResolver(()))
    logical = LogicalHistoryEvent(event.event_key,event.source,event.source_event_id,customer,(customer,),kind,
                                 event.event_datetime_utc,(HistoryVariant(event,(event.provenance,)),),1,False,False,False,event.quality)
    return HistoryPage(status,customer,(logical,),cursor,generation=generation,source_version="v")


PRODUCT = [{"ids": {"offline1C": "000123456789"}}]


@pytest.mark.parametrize("kind", [K.VIEW, K.FAVORITE])
@pytest.mark.parametrize("products", [[], [{}], [{"ids": {}}]])
def test_productless_actions_are_filtered_in_presentation(kind, products):
    display = present_history(page(kind,products),kind)
    assert display.events[0][1] == ()


@pytest.mark.parametrize("kind", [K.VIEW, K.FAVORITE])
@pytest.mark.parametrize("products", [PRODUCT, [{"ids": {"website": "unknown"}}], [{"ids": {"offline1C": "a", "kanzlerKz": "b"}}]])
def test_real_product_ids_survive_missing_photo_name_and_ambiguity(kind, products):
    rows = present_history(page(kind,products),kind).events[0][1]
    assert len(rows) == 1 and not rows[0].photo_url
    assert rows[0].values[2] == "—"
    assert rows[0].values[1] != "—"


def start(card, kind, lookup):
    _, tab, _ = card
    tab._history_lookup = lookup
    tab.history.category.select(kind)
    tab.customer_input.setText("C")
    tab.search()
    return tab, tab.history


@pytest.mark.parametrize("kind", [K.VIEW, K.FAVORITE])
def test_empty_pages_continue_to_real_products_without_duplicates(card, kind):
    calls = []
    def lookup(customer, **kwargs):
        cursor = kwargs['cursor']
        calls.append((get_ident(),cursor))
        if cursor is None:
            return page(kind,[],cursor="second",key="empty")
        return page(kind,PRODUCT,key="real")
    tab, history = start(card,kind,lookup)
    wait_for(lambda: tab.result is not None and history._active is None)
    assert [cursor for _,cursor in calls] == [None,"second"]
    assert all(thread != get_ident() for thread,_ in calls)
    assert history.table.rows.rowCount() == 1 and history.more.isHidden()
    assert history.table.empty_label.isHidden()
    history.load_more()
    assert len(calls) == 2


@pytest.mark.parametrize("kind", [K.VIEW, K.FAVORITE])
def test_empty_terminal_page_hides_more(card, kind):
    calls = []
    def lookup(customer, **kwargs):
        calls.append(kwargs['cursor'])
        return page(kind,[],cursor="last" if kwargs['cursor'] is None else None,key=str(len(calls)))
    tab, history = start(card,kind,lookup)
    wait_for(lambda: tab.result is not None and history._active is None)
    assert calls == [None,"last"] and history.table.rows.rowCount() == 0
    assert history.more.isHidden()
    assert history.table.empty_label.text() == "Нет событий в выбранной категории."


def test_many_empty_pages_are_bounded_and_explicit_continuation_remains_available(card):
    calls = []
    def lookup(customer, **kwargs):
        calls.append(kwargs['cursor'])
        n = len(calls)
        return page(K.VIEW,PRODUCT if n == MAX_EMPTY_PAGES+1 else [],cursor=str(n),key=str(n))
    tab, history = start(card,K.VIEW,lookup)
    wait_for(lambda: tab.result is not None and history._active is None)
    assert len(calls) == MAX_EMPTY_PAGES and not history.more.isHidden()
    assert history.more.text() == "Искать товары дальше"
    assert "Нет событий" not in history.table.empty_label.text()
    QTest.qWait(30)
    assert len(calls) == MAX_EMPTY_PAGES
    history.more.click()
    wait_for(lambda: history._active is None)
    assert len(calls) == MAX_EMPTY_PAGES+1 and history.table.rows.rowCount() == 1
    assert history.more.text() == "Показать ещё"


def test_repeated_cursor_stops_without_loop(card):
    calls = []
    def lookup(customer, **kwargs):
        calls.append(kwargs['cursor'])
        return page(K.VIEW,[],cursor="same",key=str(len(calls)))
    tab, history = start(card,K.VIEW,lookup)
    wait_for(lambda: tab.result is not None and history._active is None)
    assert calls == [None,"same"]
    assert history.property("historyStatus") == S.STALE_CURSOR.value
    assert history.more.isHidden()


def test_empty_page_continuation_cancels_on_category_change(card):
    entered, release = Event(), Event()
    calls = []
    def lookup(customer, **kwargs):
        kind,cursor = kwargs['kinds'][0],kwargs['cursor']
        calls.append((kind,cursor))
        if kind is K.VIEW and cursor is not None:
            entered.set()
            assert release.wait(4)
        return page(kind,PRODUCT if kind is K.FAVORITE else [],cursor="next" if kind is K.VIEW else None,key=str(len(calls)))
    tab, history = start(card,K.VIEW,lookup)
    try:
        wait_for(entered.is_set)
        previous = history._active
        history.buttons[K.FAVORITE].click()
        assert previous.cancelled.is_set()
    finally:
        release.set()
    wait_for(lambda: history._active is None)
    assert calls == [(K.VIEW,None),(K.VIEW,"next"),(K.FAVORITE,None)]
    assert history.table.rows.rowCount() == 1 and history.category.kind is K.FAVORITE


@pytest.mark.parametrize("dark", [False, True])
def test_uniform_row_background_keeps_grid_selection_copy_and_category_state(dark):
    # Qt Material repolishing after many destroyed fixture windows can crash
    # native Qt on Windows. Each theme gets a fresh QApplication/process.
    script = """import importlib, tempfile
from PyQt6.QtWidgets import QApplication
resources=importlib.import_module('qt_material.resources.generate')
resources.RESOURCES_PATH=tempfile.mkdtemp(prefix='ui06_theme_')
from tests.tabs.test_customer_ui06 import _check_uniform_theme
_check_uniform_theme(DARK)
""".replace("DARK", repr(dark))
    completed = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", script],
                               capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _check_uniform_theme(dark):
    from Application.tabs.customer_history_widgets import CustomerHistorySection
    app=QApplication([])
    apply_app_theme(app,dark)
    history=CustomerHistorySection()
    table=history.table
    table.reset_history(K.ORDER)
    table.append((HistoryRow(("","code","Название","Вид","Коллекция","Дата")),)*2)
    assert not table.alternatingRowColors() and table.showGrid()
    table.selectRow(1)
    model=table.rows
    colors=[]
    for row in (0,1):
        image=QImage(180,226,QImage.Format.Format_ARGB32)
        image.fill(QColor("white"))
        painter=QPainter(image)
        option=QStyleOptionViewItem()
        option.initFrom(table)
        from PyQt6.QtCore import QRect
        option.rect=QRect(0,0,180,226)
        if row:
            option.state |= QStyle.StateFlag.State_Selected
        table.itemDelegate().paint(painter,option,model.index(row,1))
        painter.end()
        colors.append(image.pixelColor(5,5))
    assert colors[0] == colors[1]
    history.resize(1200,650)
    table.show()
    history.show()
    QTest.qWait(40)
    canvas=table.viewport().grab().toImage()
    x=table.columnViewportPosition(1)+10
    assert canvas.pixelColor(x,table.rowViewportPosition(0)+10) == canvas.pixelColor(x,table.rowViewportPosition(1)+10)
    QTest.keyClick(table,Qt.Key.Key_C,Qt.KeyboardModifier.ControlModifier)
    assert "code" in QApplication.clipboard().text()
    assert "code" in table.selectionModel().selectedRows()[0].siblingAtColumn(1).data()
    assert history.buttons[K.ORDER].isChecked()
    history.close()


@pytest.mark.parametrize("collection,expected", [("Весна-Лето 2025","Коллекция: Весна-Лето 2025"),("","Коллекция: Не указано")])
def test_collection_is_immediately_after_code_without_score_changes(app, collection, expected):
    rec=RawBprRecommendation(1,"269245_L",1.234567890123456)
    content=ProductContent(rec,"Название",collection=collection)
    widget=ProductCard(content)
    box=widget.layout()
    assert box.indexOf(widget.collection) == box.indexOf(widget.code)+1
    assert widget.collection.text() == expected
    assert widget.content.recommendation is rec and widget.content.recommendation.raw_score == 1.234567890123456
    widget.deleteLater()



def test_automatic_empty_page_search_keeps_generation_guard(card):
    calls=[]
    def lookup(customer,**kwargs):
        calls.append(kwargs['cursor'])
        return page(K.VIEW,[] if len(calls)==1 else PRODUCT,cursor="next" if len(calls)==1 else None,
                    generation="old" if len(calls)==1 else "new")
    tab,history=start(card,K.VIEW,lookup)
    wait_for(lambda: tab.result is not None and history._active is None)
    assert calls == [None,"next"]
    assert history.property("historyStatus") == S.STALE_CURSOR.value
    assert history.table.rows.rowCount() == 0


def test_duplicate_event_page_is_skipped_without_duplicate_rows(card):
    calls=[]
    def lookup(customer,**kwargs):
        calls.append(kwargs['cursor'])
        n=len(calls)
        return page(K.VIEW,PRODUCT,key="same" if n<3 else "different",cursor=str(n) if n<3 else None)
    tab,history=start(card,K.VIEW,lookup)
    wait_for(lambda: tab.result is not None and history._active is None)
    assert history.table.rows.rowCount() == 1
    history.load_more()
    history.load_more()
    wait_for(lambda: history._active is None)
    assert calls == [None,"1","2"]
    assert history.table.rows.rowCount() == 2 and history.more.isHidden()

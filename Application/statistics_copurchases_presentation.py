"""Saved co-purchase tables, shared by GUI/XLSX/HTML; never source reads."""
from Application.statistics_cohorts_categories import UNKNOWN_CATEGORY,category_key

COPURCHASE_TITLE='Совместные покупки категорий'
COPURCHASE_LEGACY_TEXT='Нет данных о совместных покупках в сохранённом результате. Для их получения пересчитайте статистику.'


def copurchase_sections(strict,*,legacy=False):
    saved=strict.get('category_copurchases') if strict is not None else None
    titles=('Категории в составе заказов','Совместные покупки категорий','Сила сочетаний категорий')
    notes=[('Методика','Все уникальные подтверждённые заказы той же области, что и покупательская аналитика, '
        'без товарного отбора и ограничения первыми двумя покупками. Прямые категории взяты из снимка каталога расчёта; '
        'родительские категории не добавляются. Заказ учитывается один раз в категории и неупорядоченной паре; '
        'покупатель — один раз за период в каждой категории и паре.'),
        ('Доли','Знаменатель доли — все подтверждённые заказы. Суммы долей могут превышать 100%. '
         'Условные частоты показывают совместное присутствие в наблюдаемых заказах, не прогноз будущей покупки '
         'и не причинную связь. Неопределённая категория не является конкретной товарной группой.')]
    if saved is None or not saved['orders']:
        message=(COPURCHASE_LEGACY_TEXT if legacy or strict is not None and 'category_copurchases' not in strict else
                 'Нет подтверждённых заказов в доступной области наблюдения' if saved is not None else
                 'Нет доступной базы подтверждённых заказов')
        return [(COPURCHASE_TITLE,[],[],notes)] + [(title,[],[
            (['Показатель','Доступность'],[('Нет данных',message)])],[]) for title in titles]
    catalog={r['code']:r for r in saved['categories']}
    labels={code:(' / '.join(r['path']) if r['path'] else r['name']) for code,r in catalog.items()}
    frequencies={}
    for label in labels.values():frequencies[label]=frequencies.get(label,0)+1
    labels={code:(f'{label} [{code}]' if code is not None and (frequencies[label]>1 or label==UNKNOWN_CATEGORY) else label)
            for code,label in labels.items()}
    def pair_label(code):
        return labels[code]+(' — неполные данные' if code is None else '')
    notes.append(('База',f"Подтверждённых заказов: {saved['orders']} — показатель «Заказы (подтверждённые покупки)» выше. "
        f"Заказов с неопределённой категорией: {saved['unknown_orders']}. "
        'Строки таких сочетаний помечены «неполные данные».'))
    main=(COPURCHASE_TITLE,[
        ('Заказы с двумя и более категориями',saved['multicategory_orders']),
        ('Доля многокатегорийных заказов, %',saved['multicategory_rate']),
        ('Покупатели с многокатегорийными заказами',saved['multicategory_buyers']),
        ('Различных наблюдаемых пар категорий',len(saved['pairs']))],[],notes)
    rows=sorted(catalog.values(),key=lambda r:(-r['orders'],category_key(r['code'])))
    tables=[
        (['Категория','Заказов с категорией','Покупателей','Доля заказов, %'],
         [(labels[r['code']],r['orders'],r['buyers'],r['order_rate']) for r in rows]),
        (['Категория 1','Категория 2','Заказов вместе','Уникальных покупателей','Доля заказов, %'],
         [(pair_label(r['category1']),pair_label(r['category2']),r['orders'],r['buyers'],r['order_rate']) for r in saved['pairs']]),
        (['Категория 1','Категория 2','Категория 2 в заказах категории 1, %','Категория 1 в заказах категории 2, %'],
         [(pair_label(r['category1']),pair_label(r['category2']),r['given_first_rate'],r['given_second_rate']) for r in saved['pairs']])]
    return [main]+[(title,[],[table],[]) for title,table in zip(titles,tables)]

"""Small immutable-snapshot scenarios for XLSX checks; no source/database access."""

from copy import deepcopy

from Application.statistics_cache import COUNT_FIELDS, validate_result


def empty_snapshot(base, *, missing_profiles=False):
    result = deepcopy(base)
    for key in COUNT_FIELDS:
        result[key] = 0
    for key, value in result.items():
        if isinstance(value, float):
            result[key] = 0.
    result['customers'] = None if missing_profiles else 0
    for key in ('mean_age', 'median_age'):
        result[key] = None
    for key in ('purchase_quantity', 'resolved_purchase_quantity', 'mean_purchase_units_per_order', 'median_purchase_units_per_order'):
        result[key] = '0'
    for key in ('gender_distribution', 'age_distribution', 'view_user_activity_distribution', 'favorite_user_activity_distribution',
                'interaction_activity_distribution', 'purchase_order_distribution', 'purchase_basket_distribution',
                'view_availability_distribution'):
        result[key] = tuple((row[0], 0, 0.) for row in result[key])
    for key in ('action_types', 'line_statuses', 'namespaces', 'top_products', 'order_monthly_dynamics', 'store_statistics',
                'ordering_method_distribution', 'delivery_type_distribution', 'payment_type_distribution', 'action_channel_statistics',
                'action_monthly_dynamics', 'top_viewed_products', 'top_favorited_products', 'top_purchased_products',
                'product_category_statistics', 'product_gender_statistics', 'product_season_statistics', 'product_style_statistics'):
        result[key] = ()
    result['diagnostics'] = tuple((name, 0) for name, _ in result['diagnostics'])
    result['order_financials'] = tuple((currency, 0, 0, '0', '0', '0', '0') for currency in ('RUB', 'KZT'))
    result['delivery_financials'] = tuple((currency, 0, '0', '0', '0') for currency in ('RUB', 'KZT'))
    result['view_price_statistics'] = tuple((currency, 0, '0', '0') for currency in ('RUB', 'KZT'))
    result['warnings'] = ('Canonical Customers отсутствует: число профилей неизвестно.',) if missing_profiles else ()
    if result.get('product_category_paths') is not None:
        result['product_category_paths'] = ()
    return validate_result(result)


def complete_snapshot(base):
    """Two invented customers, one product, both currencies and all chart families."""
    result = deepcopy(base)
    result.update(source_actions=4, total_interactions=5, orders=2, order_lines=2, order_customers=2,
                  actions_with_product=3, favorite_interactions=1, purchase_interactions=2, purchase_quantity='8',
                  mean_interactions=2.5, median_interactions=2.5, resolved_interactions=5,
                  mean_age=35., median_age=35., favorite_users=1, purchase_users=2, view_purchase_users=1,
                  favorite_purchase_users=1, active_buyer_rate=100., purchase_orders=2,
                  resolved_favorite_interactions=1, resolved_purchase_interactions=2, resolved_purchase_quantity='8',
                  products_with_favorites=1, mean_favorites_per_user=1., median_favorites_per_user=1.)
    result['action_types'] = (('ProsmotrProduktaVApiMethod', 3), ('FavoriteSynthetic', 1))
    result['line_statuses'] = (('CP', 2, True),)
    result['namespaces'] = (('offline1C', 5, 5, 0, 100.),)
    result['top_products'] = (('000001', 'Рубашка', 2, 1, 2, 5),)
    result['gender_distribution'] = (('Мужчины', 1, 50.), ('Женщины', 1, 50.), ('Не указан', 0, 0.))
    result['age_distribution'] = tuple((name, int(name in ('26–35 лет', '36–45 лет')),
                                        50. if name in ('26–35 лет', '36–45 лет') else 0.)
                                       for name, *_ in result['age_distribution'])
    for key, chosen in (('interaction_activity_distribution', '2–5'), ('purchase_order_distribution', '1 заказ'),
                        ('purchase_basket_distribution', '1 позиция')):
        result[key] = tuple((name, 2 if name == chosen else 0, 100. if name == chosen else 0.) for name, *_ in result[key])
    result['favorite_user_activity_distribution'] = tuple((name, int(name == '1'), 100. if name == '1' else 0.)
                                                          for name, *_ in result['favorite_user_activity_distribution'])
    result['order_financials'] = (('RUB', 1, 1, '4', '40', '40', '40'), ('KZT', 1, 1, '4', '50', '50', '50'))
    result['order_monthly_dynamics'] = (('2025-01', 1, '40', 1, '50'),)
    result['store_statistics'] = (('RUB', 's1', 'Тестовый магазин RUB', 1, 1, 1, '4', '40', '40'),
                                  ('KZT', 's2', 'Тестовый магазин KZT', 1, 1, 1, '4', '50', '50'))
    for key in ('ordering_method_distribution', 'delivery_type_distribution', 'payment_type_distribution'):
        result[key] = (('Не указано', 2, 100.),)
    result['action_channel_statistics'] = (('web', 'Сайт', 2, 1, 100., 1, 1, 100.),)
    result['action_monthly_dynamics'] = (('2025-01', 2, 1, 1, 1),)
    result['top_viewed_products'] = (('000001', 'Рубашка', 2, 1, 1, 2),)
    result['top_favorited_products'] = (('000001', 'Рубашка', 1, 1, 2, 2),)
    result['top_purchased_products'] = (('000001', 'Рубашка', 2, 2, '8', 2, 1),)
    result['product_category_statistics'] = ((None, 'Не указано', 1, 2, 1, 2, '8'),)
    for key in ('product_gender_statistics', 'product_season_statistics', 'product_style_statistics'):
        result[key] = (('Не указано', 1, 2, 1, 2, '8'),)
    result['diagnostics'] = tuple((key, {'orders_unique': 2, 'unique_order_lines': 2, 'purchase_lines': 2,
                                        'mapped_view_actions': 3, 'mapped_favorite_actions': 1,
                                        'mapped_without_product': 1}.get(key, 0)) for key, _ in result['diagnostics'])
    return validate_result(result)


def quality_snapshot(base):
    """Saved aggregate example, with a tiny selected-item result; no source history."""
    result = complete_snapshot(base)
    result.update(source_actions=2292725, actions_with_product=1216544, actions_without_product=1076181)
    result['action_types'] = (('ProsmotrProdukta', 2173313), ('FavoriteSynthetic', 73978), ('Unknown', 45434))
    source = dict(mapped_view_actions=2173313, mapped_favorite_actions=73978,
                  mapped_without_product=1032982, unmapped_actions=45434)
    result['diagnostics'] = tuple((key, source.get(key, value)) for key, value in result['diagnostics'])
    return validate_result(result)


def category_snapshot(base):
    """Two directly assigned items; same leaf name, distinct codes and roots."""
    result = complete_snapshot(base)
    result.update(unique_resolved_items=2, products_with_views=2, products_with_purchases=2)
    result['top_products'] = (('000001', 'A', 1, 1, 1, 3), ('000002', 'B', 1, 0, 1, 2))
    result['top_viewed_products'] = (('000001', 'A', 1, 1, 1, 1), ('000002', 'B', 1, 1, 0, 1))
    result['top_favorited_products'] = (('000001', 'A', 1, 1, 1, 1),)
    result['top_purchased_products'] = (('000001', 'A', 1, 1, '4', 1, 1), ('000002', 'B', 1, 1, '4', 1, 0))
    result['product_category_statistics'] = (('000123', 'Рубашки', 1, 1, 1, 1, '4'),
                                             ('000124', 'Рубашки', 1, 1, 0, 1, '4'))
    result['product_category_paths'] = (('000123', (('Длинное название раздела ' * 20).strip(), 'Рубашки')),
                                        ('000124', ('Другой раздел', 'Рубашки')))
    for key in ('product_gender_statistics', 'product_season_statistics', 'product_style_statistics'):
        result[key] = (('Не указано', 2, 2, 1, 2, '8'),)
    return validate_result(result)


def customer_scope_snapshot(base, *, missing_profiles=False, empty=False):
    from Application.customer_statistics_scope import CustomerScope
    from Application.analysis_filter import AnalysisFilter
    from Application.interactions import InteractionType
    result = empty_snapshot(base, missing_profiles=missing_profiles) if missing_profiles or empty else complete_snapshot(base)
    scope = CustomerScope(True, True, AnalysisFilter('2026-01-01', '2026-08-31'))
    scope.profiles_available = not missing_profiles
    if not empty:
        if not missing_profiles:
            scope.profile('with-profile')
            scope.profile('profile-only')
        scope.action('with-profile', InteractionType.VIEW, False)
        scope.action('with-profile', InteractionType.FAVORITE, False)
        scope.action('buyer', InteractionType.VIEW, True)
        scope.action('event-only', None, False)
        scope.order('buyer', True)
        scope.order('with-profile', True)
        scope.order('orders-only', False)
    result['customer_scope'] = scope.result()
    return validate_result(result)


def stress_snapshot(base):
    """Consistent aggregate-only fixture: 72 months, 240 stores, 60 collections, Top-20."""
    result = complete_snapshot(base)
    result.update(source_actions=1440, total_interactions=1800, orders=720, order_lines=720,
                  actions_with_product=1080, actions_without_product=360, view_interactions=720,
                  favorite_interactions=360, purchase_interactions=720, purchase_quantity='2880',
                  mean_interactions=900., median_interactions=900., unique_source_products=60, unique_resolved_items=60,
                  resolved_interactions=1800, repeat_buyers=2, repeat_buyer_rate=100.,
                  mean_orders_per_buyer=360., median_orders_per_buyer=360., purchase_orders=720,
                  mean_views_per_viewer=720., median_views_per_viewer=720.,
                  mean_favorites_per_user=360., median_favorites_per_user=360., view_parameter_actions=720,
                  products_with_views=60, products_with_favorites=60, products_with_purchases=60,
                  resolved_view_interactions=720, resolved_favorite_interactions=360,
                  resolved_purchase_interactions=720, resolved_purchase_quantity='2880')
    result['action_types'] = (('ProsmotrProduktaVApiMethod', 1080), ('FavoriteSynthetic', 360))
    result['line_statuses'] = (('CP', 720, True),)
    result['namespaces'] = (('offline1C', 1800, 1800, 0, 100.),)
    result['interaction_activity_distribution'] = tuple((name, 2 if name == '101+' else 0, 100. if name == '101+' else 0.)
                                                       for name, *_ in result['interaction_activity_distribution'])
    result['purchase_order_distribution'] = tuple((name, 2 if name == '11+ заказов' else 0, 100. if name == '11+ заказов' else 0.)
                                                 for name, *_ in result['purchase_order_distribution'])
    result['purchase_basket_distribution'] = tuple((name, 720 if name == '1 позиция' else 0, 100. if name == '1 позиция' else 0.)
                                                  for name, *_ in result['purchase_basket_distribution'])
    for key, chosen in (('view_user_activity_distribution', '101+'), ('favorite_user_activity_distribution', '11+')):
        result[key] = tuple((name, int(name == chosen), 100. if name == chosen else 0.) for name, *_ in result[key])
    result['order_financials'] = (('RUB', 360, 360, '1440', '14400', '40', '40'),
                                 ('KZT', 360, 360, '1440', '18000', '50', '50'))
    months = tuple(f'{2020 + i // 12}-{i % 12 + 1:02}' for i in range(72))
    result['order_monthly_dynamics'] = tuple((month, 5, '200', 5, '250') for month in months)
    result['action_monthly_dynamics'] = tuple((month, 10, 1, 5, 1) for month in months)
    result['store_statistics'] = tuple((currency, f's{i:03}', f'{i:03} ' + 'Длинное название тестового магазина ' * 6,
                                       3, 1, 3, '12', amount, mean)
                                      for currency, amount, mean in (('RUB', '120', '40'), ('KZT', '150', '50')) for i in range(120))
    for key in ('ordering_method_distribution', 'delivery_type_distribution'):
        result[key] = (('Не указано', 720, 100.),)
    result['payment_type_distribution'] = tuple((f'Способ оплаты {i:02}', 36, 5.) for i in range(20))
    result['action_channel_statistics'] = (('web', 'Сайт', 720, 1, 100., 360, 1, 100.),)
    result['view_availability_distribution'] = tuple((name, 0 if name == 'Недоступен' else 360,
                                                      0. if name == 'Недоступен' else 50.)
                                                     for name, *_ in result['view_availability_distribution'])
    result['view_price_statistics'] = (('RUB', 360, '100.25', '100.25'), ('KZT', 0, '0', '0'))
    names = [(f'{i:06}', 'Рубашка с длинным наименованием ' * 8) for i in range(20)]
    result['top_products'] = tuple((code, name, 12, 6, 12, 30) for code, name in names)
    result['top_viewed_products'] = tuple((code, name, 12, 1, 6, 12) for code, name in names)
    result['top_favorited_products'] = tuple((code, name, 6, 1, 12, 12) for code, name in names)
    result['top_purchased_products'] = tuple((code, name, 12, 2, '48', 12, 6) for code, name in names)
    result['product_category_statistics'] = ((None, 'Не указано', 60, 720, 360, 720, '2880'),)
    for key in ('product_gender_statistics', 'product_style_statistics'):
        result[key] = (('Не указано', 60, 720, 360, 720, '2880'),)
    result['product_season_statistics'] = tuple((f'{i:02} ' + 'Длинное название коллекции ' * 5, 1, 12, 6, 12, '48') for i in range(60))
    result['diagnostics'] = tuple((key, {'orders_unique': 720, 'unique_order_lines': 720, 'purchase_lines': 720,
                                        'mapped_view_actions': 1080, 'mapped_favorite_actions': 360,
                                        'mapped_without_product': 360}.get(key, 0)) for key, _ in result['diagnostics'])
    for source in result['coverage']:
        if source['intervals']:
            source['intervals'] = (('2020-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'),)
    return validate_result(result)

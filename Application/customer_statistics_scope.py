"""Internal identity flags and aggregate-only customer coverage contract. No I/O."""

import math

from Application.analysis_filter import AnalysisFilter


class CustomerScope:
    PROFILE, ACTION, ORDER, VIEW, FAVORITE, ITEMLESS, BUYER = (1, 2, 4, 8, 16, 32, 64)

    def __init__(self, actions_available, orders_available, selection):
        self.flags = {}
        self.profiles_available = False
        self.actions_available, self.orders_available = actions_available, orders_available
        self.period = dict(start_date=selection.start_date, end_date=selection.end_date)

    def add(self, identity, flags):
        self.flags[identity] = self.flags.get(identity, 0) | flags

    def action(self, identity, kind, has_product):
        flags = self.ACTION
        if kind is not None:
            flags |= self.VIEW if kind.value == 'VIEW' else self.FAVORITE
            if not has_product:
                flags |= self.ITEMLESS
        self.add(identity, flags)

    def order(self, identity, purchase):
        self.add(identity, self.ORDER | (self.BUYER if purchase else 0))

    def profile(self, identity):
        self.add(identity, self.PROFILE)

    def result(self, check=lambda: None):
        counts = dict.fromkeys(('profiles', 'actions', 'orders', 'events', 'profile_events', 'views',
                                'favorites', 'digital', 'view_favorite', 'itemless_digital', 'buyers'), 0)
        for index, flags in enumerate(self.flags.values()):
            if index % 1024 == 0:
                check()
            for name, bit in (('profiles', self.PROFILE), ('actions', self.ACTION), ('orders', self.ORDER),
                              ('views', self.VIEW), ('favorites', self.FAVORITE),
                              ('itemless_digital', self.ITEMLESS), ('buyers', self.BUYER)):
                counts[name] += bool(flags & bit)
            counts['events'] += bool(flags & (self.ACTION | self.ORDER))
            counts['profile_events'] += bool(flags & self.PROFILE and flags & (self.ACTION | self.ORDER))
            counts['digital'] += bool(flags & (self.VIEW | self.FAVORITE))
            counts['view_favorite'] += flags & (self.VIEW | self.FAVORITE) == self.VIEW | self.FAVORITE
        full = self.profiles_available and self.actions_available and self.orders_available
        counts['without_profile'] = counts['events'] - counts['profile_events'] if self.profiles_available else None
        counts['profile_only'] = counts['profiles'] - counts['profile_events'] if full else None
        counts['identified'] = len(self.flags) if full else None
        counts['non_buyers'] = counts['identified'] - counts['buyers'] if full else None
        for names, available in ((('profiles', 'profile_events'), self.profiles_available),
                                  (('actions', 'views', 'favorites', 'digital', 'view_favorite', 'itemless_digital'), self.actions_available),
                                  (('orders', 'buyers'), self.orders_available)):
            if not available:
                counts.update((name, None) for name in names)
        total = counts['identified']
        counts['buyer_rate'] = 100 * counts['buyers'] / total if total else None
        counts['non_buyer_rate'] = 100 * counts['non_buyers'] / total if total else None
        return dict(counts, profiles_available=self.profiles_available, actions_available=self.actions_available,
                    orders_available=self.orders_available, period=self.period)


def validate_customer_scope(scope):
    if scope is None:
        return
    fields = {'profiles_available', 'actions_available', 'orders_available', 'period', 'profiles', 'actions',
              'orders', 'events', 'profile_events', 'views', 'favorites', 'digital', 'view_favorite',
              'itemless_digital', 'buyers', 'without_profile', 'profile_only', 'identified', 'non_buyers',
              'buyer_rate', 'non_buyer_rate'}
    if not isinstance(scope, dict) or set(scope) != fields:
        raise ValueError('Invalid customer scope')
    if not isinstance(scope['period'], dict) or set(scope['period']) != {'start_date', 'end_date'}:
        raise ValueError('Invalid customer scope period')
    AnalysisFilter(**scope['period'])
    availability = ('profiles_available', 'actions_available', 'orders_available')
    if any(type(scope[k]) is not bool for k in availability):
        raise ValueError('Invalid customer scope availability')
    full = all(scope[k] for k in availability)
    for names, available in ((('profiles', 'profile_events', 'without_profile'), scope['profiles_available']),
                              (('actions', 'views', 'favorites', 'digital', 'view_favorite', 'itemless_digital'), scope['actions_available']),
                              (('orders', 'buyers'), scope['orders_available']),
                              (('profile_only', 'identified', 'non_buyers'), full), (('events',), True)):
        for name in names:
            value = scope[name]
            valid = type(value) is int and value >= 0 if available else value is None
            if not valid:
                raise ValueError('Invalid customer scope count')
    s = scope
    a, o, e = s['actions'] or 0, s['orders'] or 0, s['events']
    if not max(a, o) <= e <= a + o:
        raise ValueError('Invalid customer scope union')
    if s['profiles_available'] and (not 0 <= s['profile_events'] <= min(s['profiles'], e)
                                   or s['without_profile'] != e - s['profile_events']):
        raise ValueError('Invalid profile intersection')
    if s['actions_available'] and (not 0 <= s['view_favorite'] <= min(s['views'], s['favorites'])
                                  or s['digital'] != s['views'] + s['favorites'] - s['view_favorite']
                                  or not s['itemless_digital'] <= s['digital'] <= a):
        raise ValueError('Invalid digital customer union')
    if s['orders_available'] and s['buyers'] > o:
        raise ValueError('Invalid buyer subset')
    if full and (s['profile_only'] != s['profiles'] - s['profile_events']
                 or s['identified'] != s['profiles'] + s['without_profile']
                 or s['identified'] != e + s['profile_only']
                 or s['non_buyers'] != s['identified'] - s['buyers']):
        raise ValueError('Invalid identified customer union')
    for name, numerator in (('buyer_rate', 'buyers'), ('non_buyer_rate', 'non_buyers')):
        value = s[name]
        if full and s['identified']:
            if (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100
                    or not math.isclose(value, 100 * s[numerator] / s['identified'], abs_tol=1e-9)):
                raise ValueError('Invalid customer scope percentage')
        elif value is not None:
            raise ValueError('Unavailable customer scope percentage')

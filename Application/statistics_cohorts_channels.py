"""Aggregate first/second observed order channels; no identities or source I/O."""
from collections import Counter


UNKNOWN_CHANNEL = 'Не указан'


def channel_value(external_id, name):
    """Validate prepared metadata. A missing ID never becomes a name-based ID."""
    if external_id is not None and (
            not isinstance(external_id, str) or not external_id.strip() or external_id != external_id.strip()):
        raise ValueError('Invalid purchase channel identifier')
    if name is not None and not isinstance(name, str):
        raise ValueError('Invalid purchase channel name')
    return (external_id, (name or '').strip() if external_id is not None else '')


def channel_key(external_id):
    """Stable total ordering, including the explicit unknown bucket."""
    return external_id is None, external_id or ''


class ChannelCounts:
    def __init__(self):
        self.first = Counter()
        self.second = Counter()
        self.pairs = Counter()
        self.names = {}

    def _name(self, channel):
        key, name = channel
        if key is not None and name:
            self.names[key] = min(self.names.get(key, name), name)

    def add(self, first, second):
        self._name(first)
        self.first[first[0]] += 1
        if second is not None:
            self._name(second)
            self.second[second[0]] += 1
            self.pairs[first[0], second[0]] += 1

    def merge(self, other, check):
        for target, source in ((self.first, other.first), (self.second, other.second), (self.pairs, other.pairs)):
            for key, count in source.items():
                check()
                target[key] += count
        for channel in other.names.items():
            check()
            self._name(channel)

    def result(self, check):
        buyers, repeated = sum(self.first.values()), sum(self.second.values())
        channels = []
        for key in sorted(set(self.first) | set(self.second) | {None}, key=channel_key):
            check()
            first, second = self.first[key], self.second[key]
            channels.append(dict(external_id=key, name=self.names.get(key, UNKNOWN_CHANNEL),
                first_buyers=first, first_rate=100*first/buyers if buyers else None,
                second_buyers=second, second_rate=100*second/repeated if repeated else None))
        first_repeated = Counter()
        same = other = unknown = 0
        for (a, b), count in self.pairs.items():
            check()
            first_repeated[a] += count
            if a is None or b is None:
                unknown += count
            elif a == b:
                same += count
            else:
                other += count
        transitions = []
        for (a, b), count in sorted(self.pairs.items(), key=lambda p: (-p[1], channel_key(p[0][0]), channel_key(p[0][1]))):
            check()
            transitions.append(dict(first_external_id=a, second_external_id=b, buyers=count,
                repeat_rate=100*count/repeated, first_channel_repeat_rate=100*count/first_repeated[a]))
        return dict(buyers=buyers, repeat_buyers=repeated, same_known_buyers=same,
                    other_known_buyers=other, unknown_transition_buyers=unknown,
                    same_known_rate=100*same/(same+other) if same+other else None,
                    channels=channels, transitions=transitions)

"""Bounded disk population thresholds and personal RFM over the shared core."""

from dataclasses import asdict
from decimal import Decimal
import hashlib
import json

from Application.statistics_detail_contract import PersonalRfm, RfmAvailability, RFM_WINDOWS, CURRENCIES
from Application.statistics_rfm_core import POLICIES, RfmObservation, empirical_ranks


def decimal_compare(a,b):
    x,y = Decimal(a),Decimal(b)
    return (x>y)-(x<y)


class DetailRfmStore:
    def __init__(self, db, metadata, check=lambda:None):
        self.db,self.metadata,self.check = db,metadata,check
        db.create_collation('DETAIL_DECIMAL',decimal_compare)
        db.executescript("""
            CREATE TABLE rfm_features(
              customer TEXT, policy TEXT, window TEXT, currency TEXT, payload TEXT NOT NULL,
              PRIMARY KEY(customer,policy,window,currency)) WITHOUT ROWID;
            CREATE TABLE rfm_histogram(
              policy TEXT, window TEXT, currency TEXT, component TEXT,
              value TEXT COLLATE DETAIL_DECIMAL, n INTEGER NOT NULL,
              PRIMARY KEY(policy,window,currency,component,value)) WITHOUT ROWID;
        """)
        self.thresholds = {}
        self.token = None

    def add_customer(self, customer, observations):
        for obs in observations:
            self.check()
            payload = asdict(obs)
            if obs.monetary is not None:
                payload['monetary'] = str(obs.monetary)
            self.db.execute('INSERT INTO rfm_features VALUES(?,?,?,?,?)',
                            (customer,obs.policy,obs.window,obs.currency,json.dumps(payload)))
            if not obs.in_currency_base or not obs.complete_window:
                continue
            for component,value in (('R',obs.recency if obs.recency_reliable else None),
                                    ('F',obs.frequency),('M',obs.monetary)):
                if value is not None:
                    self.db.execute("""INSERT INTO rfm_histogram VALUES(?,?,?,?,?,1)
                        ON CONFLICT(policy,window,currency,component,value) DO UPDATE SET n=n+1""",
                        (obs.policy,obs.window,obs.currency,component,str(value)))

    def finalize(self):
        for policy in POLICIES:
            for window,_ in RFM_WINDOWS:
                for currency in CURRENCIES:
                    group = {}
                    for component in ('R','F','M'):
                        self.check()
                        key = (policy,window,currency,component)
                        n = self.db.execute('SELECT COALESCE(SUM(n),0) FROM rfm_histogram WHERE policy=? AND window=? AND currency=? AND component=?',key).fetchone()[0]
                        ranks = empirical_ranks(n)
                        cuts,seen = [],0
                        cursor = self.db.execute('SELECT value,n FROM rfm_histogram WHERE policy=? AND window=? AND currency=? AND component=? ORDER BY value COLLATE DETAIL_DECIMAL',key)
                        try:
                            for value,count in cursor:
                                self.check()
                                seen += count
                                while len(cuts)<len(ranks) and seen>=ranks[len(cuts)]:
                                    cuts.append(Decimal(value) if component=='M' else int(Decimal(value)))
                        finally:
                            cursor.close()
                        group[component] = dict(buyers=n,values=tuple(cuts))
                    self.thresholds[policy,window,currency] = group
        fingerprint = [([*key],{k:dict(buyers=v['buyers'],values=[str(x) for x in v['values']]) for k,v in group.items()})
                       for key,group in self.thresholds.items()]
        self.token = hashlib.sha256(json.dumps([self.metadata.as_of_date.isoformat(),
            [[a.isoformat(),b.isoformat()] for a,b in self.metadata.orders_coverage],fingerprint],sort_keys=True).encode()).hexdigest()
        return self.thresholds

    def personal(self, customer):
        if self.token is None:
            raise ValueError('RFM thresholds are not finalized')
        cursor = self.db.execute('SELECT payload FROM rfm_features WHERE customer=? ORDER BY policy,window,currency',(customer,))
        try:
            rows = {}
            for (payload,) in cursor:
                self.check()
                data = json.loads(payload)
                if data['monetary'] is not None:
                    data['monetary'] = Decimal(data['monetary'])
                obs = RfmObservation(**data)
                cuts = self.thresholds[obs.policy,obs.window,obs.currency]
                rs,fs,ms,segment = obs.scores(*(cuts[k]['values'] for k in ('R','F','M')))
                status = (RfmAvailability.OUTSIDE_BASE if not obs.in_base else
                          RfmAvailability.NOT_IN_CURRENCY_BASE if not obs.in_currency_base else
                          RfmAvailability.UNSCORABLE if segment=='UNSCORABLE' else RfmAvailability.AVAILABLE)
                rows[obs.policy,obs.window,obs.currency] = PersonalRfm(obs.policy,obs.window,obs.currency,
                    customer,self.metadata.identity_space,self.metadata.as_of_date,self.token,status,
                    obs.recency,obs.frequency,obs.monetary,rs,fs,ms,segment,obs.recency_reliable)
            return tuple(rows[p,w,c] for p in POLICIES for w,_ in RFM_WINDOWS for c in CURRENCIES)
        finally:
            cursor.close()

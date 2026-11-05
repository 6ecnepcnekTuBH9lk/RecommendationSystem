"""Fixed segment business counters over existing RFM observations; no scoring or I/O."""
from dataclasses import dataclass
from decimal import Decimal,localcontext


@dataclass(slots=True)
class _Segment:
    buyers: int = 0
    orders: int = 0
    valued: int = 0
    amount: Decimal = Decimal(0)
    complete_buyers: int = 0
    complete_amount: Decimal = Decimal(0)


class SegmentBusiness:
    def __init__(self,segments):
        self.groups={segment:_Segment() for segment in segments}

    def add(self,segment,frequency,valued,amount,monetary):
        group=self.groups[segment]
        group.buyers+=1
        group.orders+=frequency
        group.valued+=valued
        # Preserve the caller's Decimal flags/settings used by the existing RFM.
        with localcontext():
            group.amount+=amount
            if monetary is not None:
                group.complete_buyers+=1
                group.complete_amount+=monetary

    def result(self,buyers,known_amount):
        rows=[]
        for segment,group in self.groups.items():
            with localcontext():
                share=str(100*group.amount/known_amount) if known_amount else None
                mean=str(group.amount/group.valued) if group.valued else None
                full_mean=str(group.complete_amount/group.complete_buyers) if group.complete_buyers else None
            rows.append(dict(segment=segment,buyers=group.buyers,
                buyer_rate=100*group.buyers/buyers if buyers else None,
                observed_orders=group.orders,valued_orders=group.valued,known_amount=str(group.amount),
                known_amount_rate=share,mean_known_order=mean,complete_m_buyers=group.complete_buyers,
                complete_m_amount=str(group.complete_amount),mean_complete_m=full_mean))
        return rows

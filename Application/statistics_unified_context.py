"""Explicit compatibility of a captured aggregate report and detail scope."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from Application.analysis_filter import AnalysisFilter
from Application.statistics_detail_contract import DetailPeriod, OrderMode, ORDER_COLUMNS
from Application.statistics_detail_storage import DetailExportError

@dataclass(frozen=True)
class UnifiedContext:
    period: DetailPeriod
    as_of: date

def context_for_result(result):
    selection=AnalysisFilter.from_dict(result["analysis_filter"])
    cutoffs={result[field]["as_of_date"] for field in ("rfm_statistics","cohort_statistics") if result.get(field)}
    if len(cutoffs)>1:
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    if cutoffs:
        cutoff=date.fromisoformat(cutoffs.pop())
    elif selection.end_date:
        cutoff=date.fromisoformat(selection.end_date)
    else:
        dates=[datetime.fromisoformat(b.replace("Z","+00:00"))-timedelta(microseconds=1)
               for item in result["coverage"] if item["source"]=="Orders" for _,b in item["intervals"]]
        if not dates:
            raise DetailExportError("REPORT_CONTEXT_MISMATCH")
        calculated=datetime.fromisoformat(result["calculated_at"]).astimezone(timezone.utc).date()
        cutoff=min(calculated,max(dates).date())
    period=DetailPeriod(date.fromisoformat(selection.start_date) if selection.start_date else None,
                        date.fromisoformat(selection.end_date) if selection.end_date else None)
    buyer=result.get("buyer_statistics")
    if buyer is not None and buyer["period"]!={"start_date":selection.start_date,"end_date":selection.end_date}:
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    return UnifiedContext(period,cutoff)

def validate_context(result,as_of,period,mode):
    context=context_for_result(result)
    if context.as_of!=as_of or context.period!=period or mode is not OrderMode.ALL_SELECTED:
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    return context

def _intervals(values,period,cutoff):
    start=(datetime.combine(period.start_date,datetime.min.time(),timezone.utc) if period.start_date else None)
    until=datetime.combine(cutoff+timedelta(days=1),datetime.min.time(),timezone.utc)
    if period.end_date:
        until=min(until,datetime.combine(period.end_date+timedelta(days=1),datetime.min.time(),timezone.utc))
    result=[]
    for a,b in values:
        a,b=(datetime.fromisoformat(v.replace("Z","+00:00")) if isinstance(v,str) else v for v in (a,b))
        a=max(a,start) if start else a
        b=min(b,until)
        if a<b:
            result.append((a,b))
    merged=[]
    for a,b in sorted(result):
        if merged and a<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(merged[-1][1],b))
        else:
            merged.append((a,b))
    return tuple(merged)

def validate_coverage(result,metadata):
    buyer=result.get("buyer_statistics")
    saved=buyer["coverage"] if buyer is not None else next(
        (item["intervals"] for item in result["coverage"] if item["source"]=="Orders"),())
    if _intervals(saved,metadata.event_period,metadata.as_of_date)!=_intervals(
            metadata.orders_coverage,metadata.event_period,metadata.as_of_date):
        raise DetailExportError("REPORT_SOURCE_COVERAGE_MISMATCH")

def validate_purchase_counts(result,db,control):
    """Validate prepared detail facts, without rerunning aggregate algorithms."""
    buyer=result.get("buyer_statistics")
    if buyer is None:
        return
    for policy,prefix in (("STRICT_COMPLETED","strict"),("EXTENDED_COMPLETED","extended")):
        control.check()
        accepted=ORDER_COLUMNS.index(prefix+"_accepted")
        lines=ORDER_COLUMNS.index(prefix+"_accepted_lines")
        row=db.execute("SELECT COUNT(*),COUNT(DISTINCT customer),COALESCE(SUM(CAST(json_extract(payload,?) AS INTEGER)),0) "
                       "FROM event_orders WHERE json_extract(payload,?)='1'",
                       (f"$[{lines}]",f"$[{accepted}]")).fetchone()
        saved=buyer["policies"][policy]
        if tuple(row)!=(saved["orders"],saved["buyers"],saved["positions"]):
            raise DetailExportError("REPORT_DETAIL_COUNTS_MISMATCH")


"""Synthetic demographic buyer facts: exact order medians and aggregate-only DTO."""
from copy import deepcopy
from decimal import Decimal
import json
import pytest
from Application.statistics_buyer_aggregates import POLICIES,GENDERS
from Application.customer_demographics import AGE_LABELS
from Application.statistics_buyer_validation import validate_buyer_statistics
from tests.test_statistics_buyer_aggregates import aggregate,purchase,result,AT,COVERAGE,saved_coverage


def analytics(data):
    return result(data)["policies"][POLICIES[0]]["demographic_analytics"]


def group(rows,label):
    return next(row for row in rows if row["label"]==label)


def test_gender_profiles_missing_unknown_and_repeat_frequency():
    data=aggregate([purchase("man",str(i),lines=(("F",2,str(v),"offline1C"),))
                    for i,v in enumerate((1000,2000,3000))]+[
                    purchase("woman","w"),purchase("unknown","u"),purchase("missing","m")])
    data.profile("man","male","1990-01-01",AT.date())
    data.profile("woman","female","2000-01-01",AT.date())
    data.profile("unknown",None,None,AT.date())
    a=analytics(data)
    man=group(a["gender"],"Мужчины")
    assert (man["buyers"],man["orders"],man["single_buyers"],man["repeat_buyers"])==(1,3,0,1)
    assert man["mean_orders"]==3 and man["repeat_rate"]==100 and man["buyer_rate"]==25
    assert man["money"]["RUB"]==dict(known_orders=3,known_buyers=1,amount="6000",mean_order="2000",
                                   median_order="2000",amount_rate=pytest.approx(6000/6030*100))
    unknown=group(a["gender"],"Не указан")
    assert unknown["buyers"]==unknown["orders"]==2
    assert result(data)["policies"][POLICIES[0]]["demography"]["missing_profiles"]==1
    assert [r["label"] for r in a["gender"]]==list(GENDERS)
    assert [r["label"] for r in a["age"]]==list(AGE_LABELS)
    assert "man" not in json.dumps(a) and "missing" not in json.dumps(a)


@pytest.mark.parametrize("age,label",[(17,AGE_LABELS[0]),(18,AGE_LABELS[1]),(25,AGE_LABELS[1]),
    (26,AGE_LABELS[2]),(35,AGE_LABELS[2]),(36,AGE_LABELS[3]),(45,AGE_LABELS[3]),
    (46,AGE_LABELS[4]),(55,AGE_LABELS[4]),(56,AGE_LABELS[5]),(65,AGE_LABELS[5]),(66,AGE_LABELS[6])])
def test_existing_calendar_age_bands(age,label):
    data=aggregate([purchase()])
    data.profile("buyer","male",f"{AT.year-age}-01-01",AT.date())
    rows=analytics(data)["age"]
    assert group(rows,label)["buyers"]==group(rows,label)["orders"]==1
    assert sum(r["buyers"] for r in rows)==1


def test_exact_median_is_over_orders_not_customer_turnovers_and_includes_zero():
    data=aggregate([purchase("a",str(i),lines=(("F",1,str(v),"offline1C"),)) for i,v in enumerate((1000,2000,3000))]+[
        purchase("b","b",lines=(("sold",1,"20000","offline1C"),)),
        purchase("c","c",lines=(("F",1,"0","kanzlerKz"),))])
    for customer in ("a","b","c"):data.profile(customer,"female",None,AT.date())
    money=group(analytics(data)["gender"],"Женщины")["money"]
    assert money["RUB"]["known_buyers"]==2 and money["RUB"]["known_orders"]==4
    assert Decimal(money["RUB"]["median_order"])==2500
    assert Decimal(money["RUB"]["mean_order"])==6500
    assert money["KZT"]["known_orders"]==1 and Decimal(money["KZT"]["median_order"])==0
    assert money["KZT"]["amount_rate"] is None


@pytest.mark.parametrize("values,median",[
    (("1.123456789","2.123456789","3.123456789"),"2.123456789"),
    (("1.123456789","2.123456789"),"1.623456789"),
    (("12.345","12.345","12.345","12.345"),"12.345")])
def test_decimal_order_median_is_exact(values,median):
    data=aggregate(purchase(str(i%2),str(i),lines=(("F",1,v,"offline1C"),)) for i,v in enumerate(values))
    assert Decimal(group(analytics(data)["gender"],"Не указан")["money"]["RUB"]["median_order"])==Decimal(median)


def test_money_issues_count_as_orders_and_never_become_currency_zero():
    data=aggregate([
        purchase("a","known"),purchase("a","kzt",lines=(("F",1,"100","kanzlerKz"),)),
        purchase("a","unknown",lines=(("F",1,"20","unsupported"),)),
        purchase("a","bad",lines=(("F",1,None,"offline1C"),)),
        purchase("a","mixed",lines=(("F",1,"20","offline1C"),("F",1,"30","kanzlerKz"))),
        purchase("only-bad","bad",lines=(("F",1,None,"offline1C"),))])
    data.profile("a","male",None,AT.date())
    data.profile("only-bad","female",None,AT.date())
    a=analytics(data)
    male=group(a["gender"],"Мужчины")
    assert male["orders"]==5 and male["unvalued_orders"]==3
    assert male["money"]["RUB"]["known_orders"]==male["money"]["KZT"]["known_orders"]==1
    assert male["money"]["RUB"]["amount"]=="10" and male["money"]["KZT"]["amount"]=="100"
    female=group(a["gender"],"Женщины")
    assert female["orders"]==female["unvalued_orders"]==1
    assert all(m["mean_order"] is m["median_order"] is None and m["known_orders"]==0 for m in female["money"].values())
    assert group(a["gender"],"Не указан")["buyers"]==0


def test_missing_profiles_empty_and_extended_are_independent():
    missing=result(aggregate([purchase()],profiles=False))
    assert missing["policies"][POLICIES[0]]["demography"] is None
    empty=analytics(aggregate())
    assert len(empty["gender"])==3 and len(empty["age"])==8
    assert all(r["buyers"]==r["orders"]==0 and r["mean_orders"] is None for r in empty["age"])
    value=result(aggregate([purchase(lines=(("CP",1,"20","offline1C"),))]))
    assert value["policies"][POLICIES[0]]["demographic_analytics"]["gender"][0]["buyers"]==0
    assert "demographic_analytics" not in value["policies"][POLICIES[1]]


def test_compact_money_codes_repeat_result_cancel_and_budget_guard():
    data=aggregate([purchase(key=str(i)) for i in range(2500)])
    assert data.buyers["buyer"].money_codes.itemsize==4
    assert len(data._demographic_amounts.values["RUB"])==1
    expected=analytics(data)
    assert analytics(data)==expected
    calls=0
    def stop():
        nonlocal calls
        calls+=1
        if calls==5:raise InterruptedError
    with pytest.raises(InterruptedError):data.result(AT.date(),stop)
    assert analytics(data)==expected
    data._demographic_amounts.budget.limit=data._demographic_amounts.budget.used+1024
    with pytest.raises(MemoryError):data.result(AT.date())
    limited=aggregate()
    limited._demographic_amounts.budget.limit=1
    with pytest.raises(MemoryError):limited.add(purchase())


@pytest.mark.parametrize("damage",["id","label","buyers","orders","share","repeat","known","amount","nan","mean","median","null","currency","policy"])
def test_new_block_validation_fails_closed_and_legacy_remains_valid(damage):
    value=result(aggregate([purchase(),purchase("b","b",lines=(("F",1,"20","offline1C"),))]))
    legacy=deepcopy(value)
    del legacy["policies"][POLICIES[0]]["demographic_analytics"]
    validate_buyer_statistics(legacy,coverage=saved_coverage(COVERAGE),profiles_available=True)
    p=value["policies"][POLICIES[0]];row=p["demographic_analytics"]["gender"][-1];m=row["money"]["RUB"]
    if damage=="id":row["customer_id"]="PRIVATE"
    elif damage=="label":row["label"]="Other"
    elif damage=="buyers":row["buyers"]+=1
    elif damage=="orders":row["orders"]+=1
    elif damage=="share":row["buyer_rate"]=101
    elif damage=="repeat":row["repeat_buyers"]=99
    elif damage=="known":m["known_orders"]=0
    elif damage=="amount":m["amount"]="999"
    elif damage=="nan":m["amount"]="NaN"
    elif damage=="mean":m["mean_order"]="1"
    elif damage=="median":m["median_order"]="30"
    elif damage=="null":m["median_order"]=None
    elif damage=="currency":row["money"]["USD"]=row["money"].pop("KZT")
    else:value["policies"][POLICIES[1]]["demographic_analytics"]=p["demographic_analytics"]
    with pytest.raises((ValueError,TypeError,ArithmeticError)):
        validate_buyer_statistics(value,coverage=saved_coverage(COVERAGE),profiles_available=True)



def test_observed_gap_and_selected_period_never_invent_orders():
    from datetime import datetime,timezone
    from Application.analysis_filter import AnalysisFilter
    gap=[(datetime(2025,1,1,tzinfo=timezone.utc),datetime(2025,1,2,tzinfo=timezone.utc)),
         (datetime(2026,1,1,tzinfo=timezone.utc),datetime(2026,1,3,tzinfo=timezone.utc))]
    data=aggregate([purchase("a","a","2025-01-01T12:00:00Z"),purchase("a","b")],coverage=gap)
    value=data.result(AT.date())
    validate_buyer_statistics(value,coverage=saved_coverage(gap),profiles_available=True)
    assert value["coverage_complete"] is False
    row=group(value["policies"][POLICIES[0]]["demographic_analytics"]["gender"],"Не указан")
    assert row["buyers"]==1 and row["orders"]==2 and row["mean_orders"]==2
    with pytest.raises(ValueError,match="observation"):
        data.add(purchase(stamp="2025-06-01T12:00:00Z"))
    assert data.result(AT.date())==value
    no_orders=aggregate(coverage=[]).result(AT.date())
    assert all(v is None for v in no_orders["policies"].values())
    outside=result(aggregate(selection=AnalysisFilter("2024-01-01","2024-12-31")))
    assert all(v is None for v in outside["policies"].values())


@pytest.mark.parametrize("birthday",[None,"invalid","2027-01-01"])
def test_unknown_age_keeps_profile_and_purchase(birthday):
    data=aggregate([purchase()])
    data.profile("buyer","unknown",birthday,AT.date())
    value=result(data)["policies"][POLICIES[0]]
    assert value["demography"]["missing_profiles"]==0
    assert group(value["demographic_analytics"]["age"],AGE_LABELS[-1])["orders"]==1
    assert group(value["demographic_analytics"]["gender"],GENDERS[-1])["buyers"]==1


@pytest.mark.parametrize("repeated",[False,True])
def test_cancellation_inside_both_median_paths_restores_scratch_budget(repeated):
    from array import array
    from Application.statistics_buyer_demography import _median,_Budget
    codes=array("I",[0]*5000 if repeated else range(5000))
    values=[Decimal(1)] if repeated else [Decimal(i) for i in range(5000)]
    budget=_Budget();calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==3:raise InterruptedError
    with pytest.raises(InterruptedError):_median(codes,values,check,budget)
    assert budget.used==0

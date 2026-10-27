"""Synthetic-only ML tests: never reads real canonical history or training artifacts."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from Application.model.customer_propensity_dataset import feature_schema
from Application.model.propensity_ml import artifacts, data, resources, training
from Application.model.propensity_ml.contracts import (
    COLUMNS, FEATURES, MLCancelled, MLDataError, checksum, dataset_manifest, load_json, temporal_roles, write_json,
)
from Application.model.propensity_ml.metrics import apply_sigmoid, fit_sigmoid, metrics
from Application.model.propensity_ml.preprocessing import Preprocessor, numeric_batch
from scripts.train_customer_propensity import synthetic_rows


class Guard:
    def __init__(self, cancel_after=None):
        self.calls = 0
        self.cancel_after = cancel_after

    def poll(self, **kwargs):
        self.calls += 1
        if self.cancel_after and self.calls >= self.cancel_after:
            raise MLCancelled

    def payload(self):
        return {"calls": self.calls}


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    dates = [(datetime(2022, 1, 31, tzinfo=timezone.utc) + timedelta(days=30*j)).isoformat().replace("+00:00", "Z")
             for j in range(55)]
    m = {"schema_version": 2, "version": "synthetic-v2", "features": feature_schema(),
         "target": "order_placed_within_30_days", "horizon_days": 30, "purpose": "training",
         "status": "RESEARCH_DATASET_ONLY", "as_of_level": "RETROSPECTIVE_EVENT_TIME",
         "business_validation": "REQUIRES_BUSINESS_VALIDATION", "probe": None,
         "source_version": "synthetic-source", "generation": "synthetic-generation", "identity": {"key_id":"synthetic-key-id"},
         "storage_contract_version": 1, "storage_format": "jsonl.gz",
         "compression": {"codec":"gzip","level":6,"mtime":0,"filename":""},
         "file_names": {s:s+".jsonl.gz" for s in ("train","validation","test","unknown","embargo","inference")},
         "prediction_dates": dates, "validation_start":dates[33], "test_start":dates[44],
         "label_fields_present":True, "limitations":["Synthetic, retrospective fixture only"],
         "file_sizes":{}, "checksums":{}, "counts":{"files":{},"embargo":0,"unknown":0},
         "distributions":{"by_split":{}}}
    for partition, lo, hi in (("train",0,33),("validation",33,44),("test",44,55),("unknown",0,0),("embargo",0,0),("inference",0,0)):
        rows=[]
        for j in range(lo,hi):
            for k in range(2):
                row = synthetic_rows(1)[0]
                if k:
                    for n in ("placed_orders_30d","placed_orders_90d","placed_orders_180d","placed_orders_365d",
                              "order_active_days_30d","order_active_days_90d","has_order_365d"):
                        row["features"][n]=1
                    row["features"]["days_since_last_order_365d"]=5
                row.update(customer_key=(str(k+1)*64), prediction_utc=dates[j],
                           target_until_utc=(datetime.fromisoformat(dates[j].replace("Z","+00:00"))+timedelta(days=30)).isoformat().replace("+00:00","Z"),
                           split=partition, target=k, label_state="POSITIVE" if k else "NEGATIVE", label_reasons=[])
                rows.append(row)
        file = root/m["file_names"][partition]
        file.write_bytes(gzip.compress(b"".join((json.dumps(r)+"\n").encode() for r in rows),mtime=0))
        m["file_sizes"][file.name]=file.stat().st_size
        m["checksums"][file.name]=checksum(file)
        m["counts"]["files"][partition]=len(rows)
        m["distributions"]["by_split"][partition]={"positive":len(rows)//2,"negative":len(rows)//2}
    m["counts"]["rows"]=sum(m["counts"]["files"].values())
    write_json(root/"manifest.json",m)
    return root,m


@pytest.fixture(autouse=True)
def no_acl_mutation(monkeypatch):
    monkeypatch.setattr(resources,"protect_directory",lambda path:None)
    monkeypatch.setattr(artifacts,"protect_directory",lambda path:None)


def fitted_pp():
    x=numeric_batch(synthetic_rows())
    pp=Preprocessor()
    pp.partial_fit(x,role="train")
    pp.freeze()
    return pp,x


def test_22_to_48_and_distinct_missing_meanings():
    rows=synthetic_rows(3)
    rows[1]["features"]["days_since_last_order_365d"]=0
    rows[2]["available"]["days_since_last_order_365d"]=False
    rows[2]["feature_issues"]["days_since_last_order_365d"]=["INCOMPLETE_FEATURE_COVERAGE"]
    x=numeric_batch(rows)
    j=FEATURES.index("days_since_last_order_365d")
    assert x.shape==(3,48)
    assert list(x[:,j])==[1,0,0]
    assert list(x[:,22+j])==[0,0,1]
    assert list(x[:,COLUMNS.index("semantic_null__days_since_last_order_365d")])==[1,0,0]


@pytest.mark.parametrize("role",["validation","selection","calibration","holdout","test","inference"])
def test_preprocessing_never_fit_on_future(role):
    with pytest.raises(MLDataError,match="FIT_FORBIDDEN"):
        Preprocessor().partial_fit(numeric_batch(synthetic_rows(2)),role=role)


def test_frozen_scaler_unchanged_by_future_and_reload():
    pp,x=fitted_pp()
    before=deepcopy(pp.payload())
    changed=x.copy(); changed[:,:22]*=100
    assert np.isfinite(pp.transform(changed)).all()
    assert pp.payload()==before
    assert np.array_equal(pp.transform(x),Preprocessor.from_payload(before).transform(x))
    with pytest.raises(MLDataError):pp.partial_fit(x,role="train")


@pytest.mark.parametrize("case",["order","extra","negative","null","bool","has","recency","decimal","infinite","unavailable"])
def test_bad_features_fail_closed(case):
    row=synthetic_rows(1)[0]
    if case=="order":row["features"]=dict(reversed(list(row["features"].items())))
    if case=="extra":row["features"]["future_target"]=1
    if case=="negative":row["features"]["views_30d"]=-1
    if case=="null":row["features"]["views_30d"]=None
    if case=="bool":row["features"]["views_30d"]=True
    if case=="has":row["features"]["has_order_365d"]=2
    if case=="recency":row["features"]["days_since_last_order_365d"]=365
    if case=="decimal":row["features"]["average_days_between_orders_365d"]=0.1
    if case=="infinite":row["features"]["average_days_between_orders_365d"]="NaN"
    if case=="unavailable":row["available"]["views_30d"]=False
    with pytest.raises(MLDataError):numeric_batch([row])


def test_decimal_boundary_preserves_original():
    rows=synthetic_rows(1)
    text="0.0000076736111111111111111111111111111111111111111111111111"
    rows[0]["features"]["average_days_between_orders_365d"]=text
    x=numeric_batch(rows)
    assert x[0,FEATURES.index("average_days_between_orders_365d")]>0
    assert rows[0]["features"]["average_days_between_orders_365d"]==text


def test_temporal_roles_and_overlap_rejection(dataset):
    _,m=dataset
    roles=temporal_roles(m)
    assert [len(roles[n]) for n in roles]==[33,5,3,3,11]
    changed=deepcopy(m);changed["prediction_dates"][33]="2024-10-16T00:00:00Z"
    with pytest.raises(MLDataError):temporal_roles(changed)


def test_real_streaming_reader_cache_last_batch_and_manifest(dataset,tmp_path):
    path,m=dataset;pp=Preprocessor()
    cache=data.build_cache(tmp_path/"cache",path,m,"train",m["prediction_dates"][:33],7,Guard(),preprocessor=pp)
    try:
        assert cache.x.shape==(66,48) and cache.y.sum()==33
        assert pp.train_rows==66
        loaded=data.load_cache(tmp_path/"cache",m,"train",m["prediction_dates"][:33])
        assert np.array_equal(cache.x,loaded.x);loaded.close()
    finally:cache.close()
    bad=deepcopy(m);bad["version"]="different"
    with pytest.raises(MLDataError):data.load_cache(tmp_path/"cache",bad,"train",m["prediction_dates"][:33])
    file=tmp_path/"cache/x.npy"
    with file.open("r+b") as f:f.seek(-1,2);f.write(b"X")
    with pytest.raises(MLDataError):data.load_cache(tmp_path/"cache",m,"train",m["prediction_dates"][:33])


def test_stream_corruption_and_cancel_cleanup(dataset,tmp_path):
    path,m=dataset
    with pytest.raises(MLCancelled):
        data.build_cache(tmp_path/"cancelled",path,m,"train",m["prediction_dates"][:33],7,Guard(cancel_after=4))
    assert not list(tmp_path.glob(".cache-building-*")) and not (tmp_path/"cancelled").exists()
    p=path/"train.jsonl.gz";b=p.read_bytes();p.write_bytes(b[:-1]+bytes([b[-1]^1]))
    with pytest.raises(ValueError):data.build_cache(tmp_path/"broken",path,m,"train",m["prediction_dates"][:33],7,Guard())
    assert not list(tmp_path.glob(".cache-building-*"))


def test_lock_released_after_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with data.writer_lock(tmp_path):
            with pytest.raises(MLDataError):
                with data.writer_lock(tmp_path):pass
            raise RuntimeError
    with data.writer_lock(tmp_path):pass


def test_sgd_reproducible_and_export_parity():
    pp,x=fitted_pp();x=pp.transform(x);y=np.arange(len(x))%2
    models=[]
    for _ in range(2):
        model=SGDClassifier(loss="log_loss",alpha=1e-4,average=True,random_state=123,shuffle=False)
        order=np.random.default_rng(123).permutation(len(x))
        for low in range(0,len(x),19):model.partial_fit(x[order[low:low+19]],y[order[low:low+19]],classes=[0,1])
        state=artifacts.linear_state(model,epoch=1,selection_log_loss=.1)
        np.testing.assert_allclose(model.decision_function(x),artifacts.decision(x,state),rtol=1e-6,atol=1e-6)
        models.append(state)
    assert models[0]==models[1]


def test_probability_metrics_unsigned_labels_and_empty_bins():
    y=np.array([0,1,0,1],dtype=np.uint8);p=np.array([.1,.8,.2,.9])
    result=metrics(y,p,prevalence=.5)
    assert result["log_loss"]==pytest.approx(log_loss(y,p))
    assert result["average_precision"]==1 and result["roc_auc"]==1
    assert result["constant"]["brier"]==.25
    assert any(b["rows"]==0 and b["mean_probability"] is None for b in result["reliability"])
    assert np.isfinite(metrics(y,np.array([0,1,0,1]),prevalence=.5)["log_loss"])


def test_calibration_disjoint_role_and_extreme_logits():
    z=np.linspace(-5,5,300);y=np.random.default_rng(3).binomial(1,expit(z*.5-1))
    cal=fit_sigmoid(z,y,role="calibration",guard=Guard())
    assert cal["a"]>0
    p=apply_sigmoid(np.array([-1e6,0,1e6]),cal)
    assert p[0]==0 and p[2]==1 and np.isfinite(p).all()
    for role in ("train","selection","holdout","test"):
        with pytest.raises(MLDataError):fit_sigmoid(z,y,role=role,guard=Guard())


@pytest.mark.parametrize("field,value,error",[("private_usage",4*1024**3,"PRIVATE_COMMIT"),
    ("available_commit",1,"SYSTEM_MEMORY"),("available_physical",1,"SYSTEM_MEMORY")])
def test_resource_limits(tmp_path,monkeypatch,field,value,error):
    sample={"private_usage":100,"working_set":200,"peak_working_set":300,
            "available_commit":9*1024**3,"available_physical":12*1024**3}
    sample[field]=value
    monkeypatch.setattr(resources,"volume_identity",lambda p:"same-volume")
    monkeypatch.setattr(resources.shutil,"disk_usage",lambda p:type("Disk",(),{"free":60*1024**3})())
    guard=resources.ResourceGuard([tmp_path],sampler=lambda:sample)
    with pytest.raises(MLDataError,match=error):guard.poll(force=True)


def test_disk_floor_and_manual_cancel(tmp_path,monkeypatch):
    with pytest.raises(MLCancelled):resources.ResourceGuard([tmp_path],cancelled=lambda:True).poll()
    with pytest.raises(ValueError):resources.ResourceGuard([tmp_path],disk_floor=1)
    monkeypatch.setattr(resources,"volume_identity",lambda p:"same-volume")
    monkeypatch.setattr(resources.shutil,"disk_usage",lambda p:type("Disk",(),{"free":19*1024**3})())
    sample={"private_usage":1,"peak_working_set":1,"available_physical":12*1024**3,"available_commit":9*1024**3}
    with pytest.raises(MLDataError,match="DISK_RESERVE"):
        resources.ResourceGuard([tmp_path],sampler=lambda:sample).poll()


def test_full_synthetic_pipeline_test_after_freeze_and_new_process(dataset,tmp_path,monkeypatch):
    path,m=dataset
    run=tmp_path/"run";models=tmp_path/"models";models.mkdir()
    original=data.iter_propensity_rows
    reads=[]
    def spy(directory,partition,**kwargs):
        if partition=="test":assert (run/"pipeline_frozen.json").exists()
        reads.append(partition)
        yield from original(directory,partition,**kwargs)
    monkeypatch.setattr(data,"iter_propensity_rows",spy)
    result=training.run_experiment(path,run,models,Guard(),config=training.TrainingConfig(batch_size=7,epochs=3),allow_test=True)
    assert reads==["train","validation","test"]
    assert result["test_completed"]
    model=artifacts.load_artifact(result["artifact"])
    rows=synthetic_rows(16)
    p=model.predict(rows)
    np.testing.assert_array_equal(p,np.concatenate([model.predict(rows[:3]),model.predict(rows[3:])]))
    with pytest.raises(MLDataError):model.predict(rows,feature_order=tuple(reversed(FEATURES)))
    with pytest.raises(MLDataError):model.predict(rows,schema_version=1)
    with pytest.raises(MLDataError):model.predict(rows,key_id="different")
    command=[sys.executable,"-B","-X","utf8",str(Path(__file__).resolve().parents[2]/"scripts/train_customer_propensity.py"),
             "--validate-artifact",result["artifact"]]
    child=subprocess.run(command,capture_output=True,text=True,check=True)
    assert "SYNTHETIC_ARTIFACT_VALIDATION_PASSED" in child.stdout
    checkpoint=load_json(run/"checkpoint-epoch-01.json")
    assert checkpoint["checkpoint_semantics"].startswith("SCORING_ONLY")
    target=Path(result["artifact"])/"linear.json"
    target.write_text("{}",encoding="utf-8")
    with pytest.raises(MLDataError,match="CORRUPT"):artifacts.load_artifact(result["artifact"])


def test_model_fit_test_forbidden():
    class Cache:
        manifest={"role":"test"}
    with pytest.raises(MLDataError,match="MODEL_FIT_ROLE_FORBIDDEN"):
        training.train_epochs(Cache(),Cache(),None,None,None,None,None,None)


def test_manifest_mismatch(dataset):
    p,m=dataset
    assert dataset_manifest(p)["version"]==m["version"]
    m["features"]["invented"]={"type":"integer"}
    (p/"manifest.json").write_text(json.dumps(m),encoding="utf-8")
    with pytest.raises(MLDataError):dataset_manifest(p)

"""Canonical detail ZIP backend. Explicit scope, bounded RAM, private disk state."""

from contextlib import contextmanager, closing
from dataclasses import dataclass, fields
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zipfile

from Application.paths import PROJECT_ROOT
from Application.customer_profile_service import _reader_lock, _database_version
from Application.mindbox.canonical_storage import catalog, checked_directory, effective_entries
from Application.mindbox.canonical_customers import database
from Application.mindbox.raw_reader import RawExportError, iter_export, part_files, _unique_object
from Application.mindbox.adapters import adapt_customer, adapt_order, adapt_customer_merge
from Application.mindbox.adapters._common import AdapterError, identifier, timestamp, text
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.order_dedup import OrderSnapshots, OrderSnapshotConflict
from Application.mindbox.selection import MindboxSelectionConfig
from Application.statistics_purchase_facts import PurchaseOrderFact, PurchasePolicyResult, MonetaryIssue, build_purchase_order_fact
from Application.statistics_rfm_core import RfmCustomerState
from Application.statistics_detail_contract import (
    DetailMetadata, DetailPeriod, OrderMode, OrderHeader, BuyerPolicyMetrics, CoverageStatus,
    CLIENT_COLUMNS, ORDER_COLUMNS, ORDER_LINE_COLUMNS, FORMAT_VERSION, CSV_ENCODING, CSV_DELIMITER,
    CSV_TEXT_PREFIX, NULL_TOKEN,
)
from Application.statistics_detail_projection import project_order, project_client, buyer_metrics, select_profile, csv_cells
from Application.statistics_detail_rfm import DetailRfmStore
from Application.statistics_detail_storage import (
    DetailExportError, DetailWorkspace, DetailControl, outside_repository, reject_reparse, path_identity,
)


FILES = ('clients.csv','orders.csv','order_lines.csv','manifest.json')


@dataclass(frozen=True, slots=True, repr=False)
class DetailExportResult:
    destination: Path
    manifest: dict


def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def _versions():
    package=Path(__file__).parent
    names=('statistics_detail_backend.py','statistics_detail_contract.py','statistics_detail_projection.py',
           'statistics_detail_rfm.py','statistics_detail_storage.py','statistics_rfm_core.py',
           'statistics_purchase_facts.py','statistics_purchase_policy.py','mindbox/identity.py',
           'mindbox/order_dedup.py','mindbox/adapters/orders.py','mindbox/adapters/customers.py')
    return {name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in names}


def _source_version(root, data, directories, profile_path):
    parts=[]
    for name,directory in directories:
        for path in part_files(directory,name):
            info=path.stat()
            parts.append((str(path.relative_to(root)),info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns))
    profile=_database_version(profile_path) if profile_path.exists() else None
    return hashlib.sha256(_json([data,parts,profile]).encode()).hexdigest()


@contextmanager
def canonical_source(root,check):
    """One read-only publication lock; no lock creation or journal recovery."""
    with _reader_lock(root):
        data=catalog(root)
        entries=effective_entries(data,'orders')
        merge=data['customer_merges']
        if entries and not merge:
            raise DetailExportError('CUSTOMER_MERGES_REQUIRED')
        directories=[]
        for name,entry in [('orders',entry) for entry in entries]+([('customer_merges',merge)] if merge else []):
            check()
            directory=checked_directory(root,entry['directory'],name)
            if len(part_files(directory,name)) != entry['parts']:
                raise DetailExportError('SOURCE_PARTS_MISMATCH')
            directories.append((name,directory))
        profile_path=database(root)
        before=_source_version(root,data,directories,profile_path)
        if merge:
            with closing(_checked_records('customer_merges',directories[-1][1],check)) as records:
                resolver=CustomerIdResolver(adapt_customer_merge(raw) for raw in records)
        else:
            resolver=CustomerIdResolver()
        db=None
        try:
            if profile_path.exists():
                db=sqlite3.connect(profile_path.as_uri()+'?mode=ro',uri=True,timeout=.5)
                db.execute('PRAGMA query_only=ON')
                db.execute('PRAGMA cache_size=-4096')
                db.execute('BEGIN')
                schema={row[1]:row for row in db.execute('PRAGMA table_info(profiles)')}
                if (not {'id','changed','raw'}.issubset(schema) or schema['id'][2].upper()!='TEXT'
                        or schema['id'][5]!=1 or sum(bool(r[5]) for r in schema.values())!=1):
                    raise DetailExportError('INVALID_CUSTOMERS_SCHEMA')
            yield data,entries,resolver,db,before
            check()
            if _source_version(root,catalog(root),directories,profile_path)!=before:
                raise DetailExportError('SOURCE_CHANGED')
        finally:
            if db is not None:
                db.close()


def _checked_records(name,directory,check):
    records=iter_export(name,input_dir=directory)
    try:
        for record in records:
            check()
            yield record
    finally:
        records.close()


class _DiskSnapshots:
    """Storage for the existing OrderSnapshots selector, not another dedup rule."""
    def __init__(self, db):
        self.db=db

    def get(self,key):
        row=self.db.execute('SELECT digest FROM seen WHERE order_id=?',(key,)).fetchone()
        return row[0] if row is not None else None

    def __setitem__(self,key,value):
        self.db.execute('INSERT INTO seen VALUES(?,?)',(key,value))


def _pack_fact(fact):
    policies=[]
    for p in (fact.strict,fact.extended):
        policies.append([p.line_count,str(p.units),p.currency,str(p.amount) if p.amount is not None else None,
                         sorted(issue.value for issue in p.monetary_issues)])
    return _json([fact.identity_space,fact.customer_id,fact.order_id,fact.order_datetime_utc.isoformat(),policies])


def _unpack_fact(payload):
    space,customer,order,at,policies=json.loads(payload)
    values=[PurchasePolicyResult(n,Decimal(units),currency,Decimal(amount) if amount is not None else None,
                                frozenset(MonetaryIssue(issue) for issue in issues)) for n,units,currency,amount,issues in policies]
    return PurchaseOrderFact(space,customer,order,datetime.fromisoformat(at),*values,())


def _pack_metrics(metrics):
    result=[]
    for record in metrics:
        row={}
        for f in fields(record):
            value=getattr(record,f.name)
            if isinstance(value,Enum):value=value.value
            elif isinstance(value,(datetime,date)):value=value.isoformat()
            elif isinstance(value,Decimal):value=str(value)
            row[f.name]=value
        result.append(row)
    return _json(result)


def _unpack_metrics(payload):
    result=[]
    for row in json.loads(payload):
        for key in ('units','mean_interval_days','known_amount_rub','known_amount_kzt','complete_amount_rub','complete_amount_kzt'):
            if row[key] is not None:row[key]=Decimal(row[key])
        for key in ('first_purchase_utc','last_purchase_utc'):
            if row[key] is not None:row[key]=datetime.fromisoformat(row[key])
        row['as_of_date']=date.fromisoformat(row['as_of_date'])
        row['coverage_status']=CoverageStatus(row['coverage_status'])
        result.append(BuyerPolicyMetrics(**row))
    return tuple(result)


def _schema(db):
    db.executescript("""
        CREATE TABLE seen(order_id TEXT PRIMARY KEY,digest BLOB NOT NULL) WITHOUT ROWID;
        CREATE TABLE facts(customer TEXT,ordinal INTEGER,payload TEXT NOT NULL,PRIMARY KEY(customer,ordinal)) WITHOUT ROWID;
        CREATE TABLE owners(customer TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE event_orders(ordinal INTEGER PRIMARY KEY,order_id TEXT UNIQUE,customer TEXT REFERENCES owners(customer),payload TEXT NOT NULL);
        CREATE TABLE event_lines(ordinal INTEGER REFERENCES event_orders(ordinal),line INTEGER,payload TEXT NOT NULL,PRIMARY KEY(ordinal,line)) WITHOUT ROWID;
        CREATE TABLE metrics(customer TEXT PRIMARY KEY,payload TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE profile_refs(customer TEXT REFERENCES owners(customer),source_key TEXT,PRIMARY KEY(customer,source_key)) WITHOUT ROWID;
    """)


def _ingest(root,entries,resolver,selection,db,metadata,control):
    selector=OrderSnapshots()
    selector.fingerprints=_DiskSnapshots(db)
    count=selected=lines_count=0
    for entry in entries:
        directory=checked_directory(root,entry['directory'],'orders')
        with closing(_checked_records('orders',directory,control.check)) as records:
            for raw in records:
                occurred=timestamp(raw,'firstAction.dateTimeUtc')
                if occurred >= metadata.cutoff_exclusive or not any(a <= occurred < b for a,b in metadata.effective_coverage):
                    continue
                if not selector.accept(raw):  # 01A strict conflict policy; identical repeats only.
                    continue
                source=identifier(raw,'customer.ids.mindboxId')
                customer=resolver.resolve(source)
                order_id=identifier(raw,'ids.mindboxId')
                lines=adapt_order(raw,resolver,product_namespaces=selection.order_product_namespaces)
                fact=build_purchase_order_fact(lines,identity_space=metadata.identity_space,customer_id=customer,
                    order_id=order_id,order_datetime_utc=occurred,check=control.check)
                count+=1
                payload=_pack_fact(fact)
                control.consume(len(payload.encode()))
                db.execute('INSERT INTO facts VALUES(?,?,?)',(customer,count,payload))
                if metadata.includes_event(occurred):
                    header=OrderHeader(order_id,customer,source,occurred,
                        identifier(raw,'firstAction.channel.ids.externalId'),text(raw,'firstAction.channel.name',required=True),
                        identifier(raw,'ids.retailOrderId',required=False))
                    bundle=project_order(fact,lines,metadata,header=header,check=control.check)
                    if bundle is not None:
                        db.execute('INSERT OR IGNORE INTO owners VALUES(?)',(customer,))
                        payload=_json(tuple(csv_cells(bundle.order)))
                        control.consume(len(payload.encode()))
                        db.execute('INSERT INTO event_orders VALUES(?,?,?,?)',(count,order_id,customer,payload))
                        selected+=1
                        for line in bundle.lines:
                            payload=_json(tuple(csv_cells(line)))
                            control.consume(len(payload.encode()))
                            db.execute('INSERT INTO event_lines VALUES(?,?,?)',(count,line.line_ordinal,payload))
                            lines_count+=1
                if count%256==0:
                    db.commit()
                    control.stage('history',count)
    db.commit()
    return dict(history_orders=count,orders=selected,order_lines=lines_count,identical_snapshots=selector.identical)


def _prepare_customers(db,profile_db,resolver,metadata,control):
    store=DetailRfmStore(db,metadata,control.check)
    processed=0
    cursor=db.execute('SELECT customer FROM facts GROUP BY customer ORDER BY customer')
    try:
        for (customer,) in cursor:
            state=RfmCustomerState(metadata.as_of_date,metadata.orders_coverage,identity_space=metadata.identity_space,check=control.check)
            def records():
                rows=db.execute('SELECT payload FROM facts WHERE customer=? ORDER BY ordinal',(customer,))
                try:
                    for (payload,) in rows:
                        control.check()
                        fact=_unpack_fact(payload)
                        state.add(fact)
                        yield fact
                finally:
                    rows.close()
            with closing(records()) as customer_facts:
                metrics=buyer_metrics(customer,customer_facts,metadata,check=control.check)
            db.execute('INSERT INTO metrics VALUES(?,?)',(customer,_pack_metrics(metrics)))
            store.add_customer(customer,state.observations())
            processed+=1
            if processed%256==0:
                db.commit()
                control.stage('personal_metrics',processed)
    finally:
        cursor.close()
    control.stage('rfm')
    store.finalize()
    if profile_db is not None:
        cursor=profile_db.execute('SELECT id FROM profiles ORDER BY id')
        try:
            for (source_key,) in cursor:
                control.check()
                canonical=resolver.resolve(source_key)
                if db.execute('SELECT 1 FROM owners WHERE customer=?',(canonical,)).fetchone():
                    db.execute('INSERT INTO profile_refs VALUES(?,?)',(canonical,source_key))
        finally:
            cursor.close()
    db.commit()
    return store


def _profile_candidates(customer,db,profile_db,resolver,control):
    cursor=db.execute('SELECT source_key FROM profile_refs WHERE customer=? ORDER BY source_key',(customer,))
    try:
        for (key,) in cursor:
            control.check()
            row=profile_db.execute('SELECT changed,raw FROM profiles WHERE id=?',(key,)).fetchone()
            if row is None:
                raise DetailExportError('CUSTOMERS_CHANGED')
            changed,raw=row
            record=adapt_customer(json.loads(raw,parse_float=Decimal,object_pairs_hook=_unique_object),resolver)
            if record.customer_id!=customer or record.change_datetime_utc!=timestamp({'at':changed or None},'at',required=False):
                raise DetailExportError('INCONSISTENT_PROFILE_IDENTITY')
            yield record
    finally:
        cursor.close()


class _CsvSink:
    def __init__(self,stream,control):
        self.stream,self.control,self.first=stream,control,True

    def write(self,text):
        self.control.check()
        data=text.encode('utf-8')
        if self.first:
            data=b'\xef\xbb\xbf'+data
            self.first=False
        self.control.consume(len(data))
        self.stream.write(data)
        return len(text)


def _csv_entry(archive,name,headers,rows,control):
    count=0
    info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED
    with archive.open(info,'w',force_zip64=True) as stream:
        writer=csv.writer(_CsvSink(stream,control),delimiter=CSV_DELIMITER,lineterminator='\r\n')
        writer.writerow(headers)
        with closing(rows):
            for row in rows:
                control.check()
                if len(row)!=len(headers):raise DetailExportError('CSV_SCHEMA_MISMATCH')
                writer.writerow(row)
                count+=1
                if count%256==0:
                    control.stage({'clients.csv':'clients','orders.csv':'orders','order_lines.csv':'order_lines'}[name],count)
    return count


def _clients(db,profiles,resolver,metadata,rfm,control,quality):
    cursor=db.execute('SELECT o.customer,m.payload FROM owners o JOIN metrics m ON m.customer=o.customer ORDER BY o.customer')
    try:
        for customer,payload in cursor:
            control.check()
            with closing(_profile_candidates(customer,db,profiles,resolver,control)) as candidates:
                selected=select_profile(customer,candidates,source_available=profiles is not None,check=control.check)
            metrics=_unpack_metrics(payload)
            row=project_client(customer,selected,metadata,strict=metrics[0],extended=metrics[1],
                               rfm=rfm.personal(customer),check=control.check)
            quality[selected.status.value]=quality.get(selected.status.value,0)+1
            for cell in row.rfm:
                if cell.status.value=='UNSCORABLE':
                    reason=('INCOMPLETE_WINDOW' if not rfm.thresholds[cell.policy,cell.window,cell.currency]['F']['values']
                            else 'UNRELIABLE_RECENCY' if not cell.recency_reliable else 'INCOMPLETE_MONEY')
                    quality[reason]=quality.get(reason,0)+1
            yield tuple(csv_cells(row))
    finally:
        cursor.close()


def _stored_rows(db,table):
    sql={'event_orders':'SELECT payload FROM event_orders ORDER BY ordinal',
         'event_lines':'SELECT payload FROM event_lines ORDER BY ordinal,line'}[table]
    cursor=db.execute(sql)
    try:
        for (payload,) in cursor:
            yield json.loads(payload)
    finally:
        cursor.close()


def _verify_zip(path,control):
    with zipfile.ZipFile(path) as archive:
        if tuple(archive.namelist())!=FILES:
            raise DetailExportError('ARCHIVE_MEMBERS_MISMATCH')
        for name in FILES:
            with archive.open(name) as stream:
                while stream.read(64*1024):control.check()


def _target_version(path):
    if not path.exists():return None
    info=path.stat()
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns


def export_detail_statistics(*, raw_root, destination, temporary_root, as_of_date,
                             event_period=DetailPeriod(), order_mode=OrderMode.ALL_SELECTED,
                             progress=None, cancelled=None, reserve_bytes=128*1024**2,
                             max_temporary_bytes=8*1024**3):
    """Write four linked entries atomically. No Qt, INDEXED, network or implicit as_of.

    temporary_root must be an existing admitted private directory on D: outside
    all repositories. Only synthetic sources should be used in automated tests.
    """
    workspace=None
    try:
        root=reject_reparse(raw_root).resolve()
        destination=reject_reparse(destination).absolute()
        temporary_root=reject_reparse(temporary_root).absolute()
        if (not destination.parent.is_dir() or destination.suffix.lower()!='.zip' or destination.is_dir()
                or not outside_repository(destination.parent) or destination.is_relative_to(root)
                or temporary_root.is_relative_to(root) or destination.is_relative_to(PROJECT_ROOT)):
            raise DetailExportError('INVALID_EXPORT_PATH')
        target_before=_target_version(destination)
        control=DetailControl(temporary_root,destination,progress=progress,cancelled=cancelled,
                              reserve_bytes=reserve_bytes,max_temporary_bytes=max_temporary_bytes)
        control.stage('source_validation')
        workspace=DetailWorkspace(temporary_root,destination,control)
        db=workspace.prepare()
        _schema(db)
        versions=_versions()
        with canonical_source(root,control.check) as (data,entries,resolver,profiles,token):
            selection=MindboxSelectionConfig(**data['selection'])
            coverage=tuple((entry['since'],entry['until']) for entry in entries)
            metadata=DetailMetadata(as_of_date,event_period,'REFERENCE:detail:'+token,coverage,order_mode)
            control.stage('history')
            totals=_ingest(root,entries,resolver,selection,db,metadata,control)
            control.stage('personal_metrics')
            rfm=_prepare_customers(db,profiles,resolver,metadata,control)
            expected_clients=db.execute('SELECT COUNT(*) FROM owners').fetchone()[0]
            if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
                raise DetailExportError('BROKEN_DETAIL_RELATIONS')
            quality={}
            with workspace.archive_stream() as stream:
                with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
                    control.stage('clients')
                    clients=_csv_entry(archive,FILES[0],CLIENT_COLUMNS,_clients(db,profiles,resolver,metadata,rfm,control,quality),control)
                    control.stage('orders')
                    orders=_csv_entry(archive,FILES[1],ORDER_COLUMNS,_stored_rows(db,'event_orders'),control)
                    control.stage('order_lines')
                    lines=_csv_entry(archive,FILES[2],ORDER_LINE_COLUMNS,_stored_rows(db,'event_lines'),control)
                    if (clients,orders,lines)!=(expected_clients,totals['orders'],totals['order_lines']):
                        raise DetailExportError('ROW_COUNT_MISMATCH')
                    manifest=dict(format_version=FORMAT_VERSION,generated_at=datetime.now(timezone.utc).isoformat(),
                        as_of_date=as_of_date.isoformat(),event_period=dict(start_date=event_period.start_date.isoformat() if event_period.start_date else None,
                            end_date=event_period.end_date.isoformat() if event_period.end_date else None),
                        source_coverage=[[a.isoformat(),b.isoformat()] for a,b in metadata.orders_coverage],
                        effective_coverage=[[a.isoformat(),b.isoformat()] for a,b in metadata.effective_coverage],
                        coverage_status=metadata.coverage_status.value,order_mode=order_mode.value,
                        identity_coverage=[data['customer_merges']['since'],data['customer_merges']['until']] if data['customer_merges'] else [],
                        identity_complete_for_orders=all(data['customer_merges'] is not None and
                            datetime.fromisoformat(data['customer_merges']['since']) <= a and b <= datetime.fromisoformat(data['customer_merges']['until'])
                            for a,b in metadata.effective_coverage),
                        profile_semantics='CURRENT_SNAPSHOT_NOT_HISTORICAL',profiles_available=profiles is not None,
                        policies=dict(STRICT_COMPLETED=['F','sold'],EXTENDED_COMPLETED=['F','sold','CP']),
                        files=list(FILES),rows=dict(clients=clients,orders=orders,order_lines=lines),
                        history=totals,quality=quality,source_token=token,rfm_population_token=rfm.token,
                        algorithms=versions,columns={'clients.csv':list(CLIENT_COLUMNS),'orders.csv':list(ORDER_COLUMNS),'order_lines.csv':list(ORDER_LINE_COLUMNS)},
                        csv=dict(encoding=CSV_ENCODING,delimiter=CSV_DELIMITER,newline='CRLF',null=NULL_TOKEN,text_prefix=CSV_TEXT_PREFIX,
                                 decimal='fixed-point exact',datetime='ISO 8601 UTC',boolean='1/0'),
                        notes=['Contains personal data. Do not commit or log CSV rows.',
                               'IDs are original strings; remove exactly one text prefix according to schema.',
                               'RUB and KZT are separate. Known subtotals are not net revenue or LTV.',
                               'First observed purchase is not proven lifetime first purchase.',
                               'Customers are current snapshot; communication permissions are UNKNOWN.',
                               'Incomplete monetary remains NULL in RFM; full-history known subtotals are separate columns.'])
                    control.stage('zip_verification')
                    payload=_json(manifest).encode('utf-8')
                    control.consume(len(payload))
                    info=zipfile.ZipInfo(FILES[3],date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                    archive.writestr(info,payload)
                os.fsync(stream.fileno())
            _verify_zip(workspace.partial,control)
            if _versions()!=versions:raise DetailExportError('ALGORITHM_CHANGED')
        workspace.cleanup_state()  # Cleanup failure must not replace an existing destination.
        control.stage('atomic_save')
        workspace.validate_partial()
        reject_reparse(destination)
        if path_identity(destination.parent)!=workspace.parent_id or _target_version(destination)!=target_before:
            raise DetailExportError('DESTINATION_CHANGED')
        os.replace(workspace.partial,destination)
        workspace.partial=None
        control.partial=None
        return DetailExportResult(destination,manifest)
    except InterruptedError:
        raise InterruptedError('Detail export cancelled') from None
    except DetailExportError:
        raise
    except OrderSnapshotConflict:
        raise DetailExportError('ORDER_SNAPSHOT_CONFLICT') from None
    except (AdapterError, RawExportError):
        raise DetailExportError('SOURCE_DATA_INVALID') from None
    except Exception as error:
        if workspace is not None and workspace.control.stopped:
            raise InterruptedError('Detail export cancelled') from None
        if isinstance(error, PermissionError):
            raise DetailExportError('ACCESS_DENIED') from None
        if isinstance(error, OSError):
            import errno
            raise DetailExportError('INSUFFICIENT_SPACE' if error.errno == errno.ENOSPC else 'IO_ERROR') from None
        raise DetailExportError() from None
    finally:
        if workspace is not None:
            try:
                workspace.cleanup_state()
            finally:
                workspace.cleanup_partial()

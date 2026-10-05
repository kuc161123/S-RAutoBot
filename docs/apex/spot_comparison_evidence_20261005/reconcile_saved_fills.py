"""Independently reconcile saved research fills to coins, cash and equity."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import pandas as pd

ROOT = Path('docs/apex/spot_comparison_evidence_20261005')
report = json.loads((ROOT / 'report.json').read_text())
assert report['complete'] and report['protocol_unchanged'] and report['code_unchanged']
audit = {'complete': True, 'arms': []}

def equal(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7), (a, b)

for row in report['rows']:
    blob = (ROOT / row['evidence_file']).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == row['evidence_sha256']
    r = json.loads(gzip.decompress(blob))
    cutoff = pd.Timestamp(r['entry_cutoff']).timestamp()
    end = pd.Timestamp(r['end']).timestamp()
    assert r['replayed_through'] == r['end']
    cash, inventory_value = r['initial_equity'], 0.0
    fees, funding, closed_pnl, wins, losses = 0.0, 0.0, 0.0, 0, 0
    counts = {'filled': 0, 'open': 0, 'closed': 0, 'expired': 0}
    for source in list(r['source'].values()) + list(r['funding_source'].values()):
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    for t in r['trades']:
        assert t['side'] == 'Buy' and t['created_at'] < cutoff
        if t['status'] == 'EXPIRED':
            counts['expired'] += 1
        if t['opened_at'] is None:
            assert not t['fills'] and t['fees'] == t['net_pnl'] == 0
            continue
        counts['filled'] += 1
        assert t['decision_at'] <= t['opened_at'] < cutoff
        coin, gross, paid = 0.0, 0.0, 0.0
        for i, f in enumerate(t['fills']):
            assert t['opened_at'] <= f['time'] <= end
            if f['reason'] == 'ENTRY':
                assert i == 0
                gross_qty = f['qty'] / (1 - r['fee_rate']) if r['market'] == 'spot' else f['qty']
                expected_fee = (gross_qty - f['qty']) * f['price'] if r['market'] == 'spot' else f['qty'] * f['price'] * r['fee_rate']
                equal(gross_qty, f['gross_purchase_qty'])
                if r['market'] == 'spot':
                    equal(gross_qty - f['fee_base_qty'], f['qty'])
                    cash -= gross_qty * f['price']
                else:
                    cash -= f['qty'] * f['price'] + expected_fee
                coin += f['qty']
            else:
                expected_fee = f['qty'] * f['price'] * r['fee_rate']
                coin -= f['qty']
                assert coin >= -1e-8
                gross += (f['price'] - t['entry']) * f['qty']
                cash += f['qty'] * f['price'] - expected_fee
            equal(expected_fee, f['fee'])
            paid += expected_fee
        equal(coin, t['remaining'])
        equal(gross, t['gross_pnl'])
        equal(paid, t['fees'])
        expected_funding = 0.0
        if r['market'] == 'perpetual':
            frame = pd.read_parquet(r['funding_source'][t['symbol']]['path'])
            for stamp, rate in frame[['ts_ms','funding_rate']].itertuples(index=False, name=None):
                when = stamp / 1000
                if not t['opened_at'] < when <= end or (t['closed_at'] is not None and when >= t['closed_at']):
                    continue
                qty = t['qty'] - sum(f['qty'] for f in t['fills'] if f['reason'] != 'ENTRY' and f['time'] <= when)
                expected_funding -= qty * t['entry'] * rate
        equal(expected_funding, t['funding'])
        equal(gross - paid + expected_funding, t['net_pnl'])
        cash += expected_funding
        fees += paid
        funding += expected_funding
        if t['status'] == 'OPEN':
            counts['open'] += 1
            price_frame = pd.read_parquet(r['source'][t['symbol']]['path'])
            last = price_frame.loc[price_frame.start < pd.Timestamp(r['end'])].iloc[-1].close
            inventory_value += t['remaining'] * last
        elif t['status'] == 'CLOSED':
            counts['closed'] += 1
            equal(coin, 0)
            closed_pnl += t['net_pnl']
            wins += t['net_pnl'] > 1e-9
            losses += t['net_pnl'] < -1e-9
    for name, number in counts.items():
        assert r[name] == number
    equal(cash, r['final_account']['cash'])
    equal(cash + inventory_value, r['final_account']['equity'])
    equal(closed_pnl, r['closed_net'])
    equal(fees, r['total_fees'])
    equal(funding, r['funding_estimate'])
    assert r['wins'] == wins and r['losses'] == losses
    assert r['minimum_available_cash'] >= -1e-7
    assert r['final_account']['pending_count'] == 0
    audit['arms'].append({'market':r['market'], 'reconstructed_cash':cash, 'reconstructed_equity':cash+inventory_value, 'fees':fees, 'funding_estimate':funding, 'counts':counts, 'checks':'fill quantity/coin fees/cash conservation/equity/funding chronology/cutoff/source hashes all passed'})
(ROOT / 'accounting_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit, indent=2))

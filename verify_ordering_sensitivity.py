"""How much of the base-vs-trailing gap is just intra-hour ordering noise?

Many signals share an entry hour. run_simulation processes them in row order, and that
order decides which ones consume the net-directional and gross-open-risk caps before the
rest get blocked. The order is arbitrary — it falls out of whatever sequence the universe
builder happened to emit. If shuffling ties moves the final balance as much as the
measured edge, the edge is not measurable at this precision.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import multiprocessing as mp

START=1500.0; BASE_RT=2*(0.0006+0.0003)
END=pd.Timestamp('2026-07-25'); W=END-pd.DateOffset(months=15)
_P=None;_CHOP=None;_D=None

def _init():
    global _P,_CHOP,_D
    import backtest_production_correct as P
    from backtest_shadow_gate import LIVE as LIVE_KW
    _P=(P,LIVE_KW)
    d=pd.read_parquet(str(__import__('pathlib').Path(__file__).resolve().parent / 'trail_5m_universe.parquet'))
    d['entry_time']=pd.to_datetime(d['entry_time'])
    _D=d[(d.entry_time>=W)&(d.entry_time<END)].reset_index(drop=True)
    _CHOP=P.load_chop_data(sorted(_D.symbol.unique()))

def run(args):
    nm,seed=args
    P,LIVE_KW=_P
    d=_D
    out=d[['entry_time','entry_price','sl_price','side','symbol','btc_bull','btc_impulse']].copy()
    out['exit_time']=pd.to_datetime(d['x_'+nm]); out['r_result']=d['r_'+nm]
    out=out.dropna(subset=['exit_time','r_result'])
    if seed>=0:                                  # shuffle, then stable-sort by hour
        out=out.sample(frac=1.0,random_state=seed)
    out=out.sort_values('entry_time',kind='mergesort').reset_index(drop=True)
    P.ROUND_TRIP_COST=BASE_RT; P.STARTING_BALANCE=START
    kw=dict(LIVE_KW,starting_balance=START,btc_bull_col='btc_bull',btc_short_col='btc_impulse')
    r=P.run_simulation(out,_CHOP,**kw)
    return nm,seed,r['final_effective'],r['max_dd_pct'],len(r['entered_trades'])

if __name__=='__main__':
    jobs=[(nm,s) for nm in ('base','s3_a1') for s in range(-1,12)]
    res={}
    with mp.Pool(max(1,mp.cpu_count()-1),initializer=_init) as pool:
        for nm,s,f,dd,n in pool.imap_unordered(run,jobs):
            res.setdefault(nm,[]).append((s,f,dd,n))
    print('='*78)
    print('INTRA-HOUR ORDERING SENSITIVITY — 15 months, $1,500, 12 shuffles + original')
    print('='*78)
    print(f"  {'rule':<8}{'original $':>13}{'median $':>12}{'min $':>12}{'max $':>12}{'spread':>9}")
    stats={}
    for nm in ('base','s3_a1'):
        rows=sorted(res[nm]); orig=[f for s,f,_,_ in rows if s==-1][0]
        sh=[f for s,f,_,_ in rows if s>=0]
        stats[nm]=(orig,np.median(sh),min(sh),max(sh))
        print(f"  {nm:<8}{orig:>13,.0f}{np.median(sh):>12,.0f}{min(sh):>12,.0f}"
              f"{max(sh):>12,.0f}{(max(sh)/min(sh)-1):>9.0%}")
    b,s=stats['base'],stats['s3_a1']
    print()
    print(f"  measured edge (original rows) : {s[0]-b[0]:+,.0f}  ({s[0]/b[0]-1:+.0%})")
    print(f"  edge on medians               : {s[1]-b[1]:+,.0f}  ({s[1]/b[1]-1:+.0%})")
    print(f"  base alone varies by          : {b[3]-b[2]:,.0f} across shuffles")
    print(f"  s3_a1 alone varies by         : {s[3]-s[2]:,.0f} across shuffles")
    # paired: same seed both arms
    bd={s_:f for s_,f,_,_ in res['base'] if s_>=0}
    sd={s_:f for s_,f,_,_ in res['s3_a1'] if s_>=0}
    d=[sd[k]-bd[k] for k in sorted(bd)]
    wins=sum(1 for x in d if x>0)
    print(f"\n  PAIRED per seed (same ordering both arms):")
    print(f"    s3_a1 beat base in {wins}/{len(d)} shuffles")
    print(f"    median paired diff {np.median(d):+,.0f}   range {min(d):+,.0f} .. {max(d):+,.0f}")

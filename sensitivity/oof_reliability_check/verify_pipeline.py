"""Integrity checks, not new research performance evidence."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_oof_reliability as r


def checks(pred_cache=None):
    # BA accounting must remain correct when classes are extremely imbalanced
    # and annotated rows have nonzero reward but zero evaluation weight.
    y=np.array([0]*31+[1]*3+[2]*2)
    b=y.copy(); e=y.copy()
    b[[0,1,31,34]]=1-b[[0,1,31,34]]
    e[[2,3,32,35]]=(e[[2,3,32,35]]+1)%3
    gate=b.copy(); gate[[0,2,31,32]]=e[[0,2,31,32]]
    ev=np.ones(len(y),bool); ev[0]=False
    scored=r.score_predictions({'base':b,'fusion':e,'gate':gate},y,ev)
    observed=scored['gate']['ba']-scored['base']['ba']
    accounted=scored['gate']['recover_ba']-scored['gate']['harm_ba']
    assert abs(observed-accounted)<1e-12
    assert np.array_equal(r.utility(np.array([0,1,2,3]),np.array([0,0,1,3]),np.array([1,1,0,3])),[-1,1,0,0])
    report={'ba_gain_equals_recover_minus_harm':True,'both_wrong_has_zero_utility':True}
    if pred_cache is not None:
        path=Path(pred_cache)/'Houston.npz'
        with np.load(path,allow_pickle=False) as a:
            d={k:a[k] for k in a.files}
        q=d['qpos']; ann=np.random.default_rng(0).choice(q,64,replace=False)
        first=r.reference_predictions(d,ann)
        changed=dict(d)
        changed['y_ids']=d['y_ids'].copy()
        query=np.setdiff1d(q,ann)
        changed['y_ids'][query]=(changed['y_ids'][query]+1)%8
        second=r.reference_predictions(changed,ann)
        assert all(np.array_equal(first[k],second[k]) for k in first)
        report['real_cache_query_label_perturbation_keeps_predictions']=True
        model=r.fit_reliability(d['base_logits'][d['spos']],d['expert_logits'][d['spos']],
                                d['y_ids'][d['spos']],d['ports_all'][d['spos']])
        local=np.flatnonzero(np.isin(q,ann))
        choices,_=r.reliability_predictions(model,d['base_logits'][q],d['expert_logits'][q],local,d['y_ids'][q[local]])
        observed=np.bincount(d['y_ids'][ann],minlength=8)>0
        bq=d['base_logits'][q].argmax(1); eq=d['expert_logits'][q].argmax(1)
        for p in choices.values():
            assert np.array_equal(p[~observed[eq]],bq[~observed[eq]])
        report['real_cache_learned_gates_observed_guard']=True
        # Cache provenance rejection: never accept partial or a mismatched parent.
        for tag,metadata in [('partial',{'schema':'oof-reliability-v1','complete':False}),
                              ('wrong_parent',{'schema':'oof-reliability-v1','complete':True,
                                               'outer_port':'Houston','outer_cache_sha256':'wrong'})]:
            with tempfile.TemporaryDirectory() as td:
                fixture=Path(td)/'fixture.npz'
                np.savez(fixture,source_rows=d['spos'],oof_base_logits=d['base_logits'][d['spos']],
                         oof_expert_logits=d['expert_logits'][d['spos']],source_labels=d['y_ids'][d['spos']],
                         source_ports=d['ports_all'][d['spos']],metadata_json=json.dumps(metadata))
                rejected=False
                try:
                    r.load_oof(fixture,d,path,'Houston')
                except ValueError:
                    rejected=True
                assert rejected
                report[tag+'_oof_archive_rejected']=True
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pred-cache',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        report=checks(a.pred_cache)
    print(json.dumps(report,indent=2))
    if a.out:
        a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')

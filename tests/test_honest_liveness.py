"""Honest runs across configurations, skipped round numbers and the record budget: no abort, strike, burn or
audit mismatch expected."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
from fedproof import model as M, protocol as PR
from test_protocol import build
cases = [dict(), dict(K=9, kp=2, Tn=2, A_min=3), dict(N=4, K=8, kp=3, Tn=1, A_min=4), dict(K=10, kp=1, Tn=3, A_min=2)]
for kw in cases:
    for rounds, mu in (((1, 2, 3), None), ((2, 5, 6, 9), None), ((1, 2, 3, 4), 2)):
        sysm, sites, nodes, log, _ = build(**kw)
        if mu is not None:
            sysm.max_uses = mu; sysm.gen = sysm.genesis()
        st = M.init_model(sysm.cfg); reps = []
        for t in rounds:
            st, rep = PR.run_round(sysm, sites, nodes, log, st, t); reps.append(rep['aborted'])
        s2, out = PR.audit(sysm.gen, log.entries)
        same = M.model_digest(sysm.cfg, s2) == M.model_digest(sysm.cfg, st)
        held = [nd._secrets_held() for nd in nodes if nd._secrets_held() != dict(round_keys=[], share_rounds=[])]
        assert reps == [None] * len(rounds) and not sysm.node_burned and not any(sysm.site_strikes.values()) and same and not held, (kw, rounds, mu)
        print(kw, rounds, 'max_uses', mu, '| aborts', reps, '| burned', sorted(sysm.node_burned), '| strikes', sum(sysm.site_strikes.values()),
              '| audit same', same, '| secrets held', held, '| valid', [o['valid'] for o in out])
print('honest liveness across configurations passed')

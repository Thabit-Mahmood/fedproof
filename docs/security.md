# Security theorems and proofs

The theorems behind the paper's security analysis and how they are proved. Statements refer to the code (fedproof/) and
to specification.md. Our own steps are argued in full; standard results are cited. Nothing here is claimed beyond what
the tests support.

## 0. Model and notation

Parties: N plants (sites), K committee nodes, one relay (the round driver), auditors. The registry (sites' and nodes'
ML-DSA-65 keys, nodes' beacon-chain commitments, every site's data and first noise commitments) is fixed at setup.

Bulletin: an append-only sequence of byte strings. Every honest party reads the same sequence in the same order
(consistency; a deployment gets it from a transparency log with gossip). The relay chooses
what is appended and when: it may drop, delay, duplicate, reorder and pad with junk, but never rewrite.

Adversary A: controls the relay; at most Tn nodes per round (a node whose signing key has ever been seized counts as
corrupted in every later round); any set of sites. A may call every public request of every honest party at any time
with any arguments (Site: round_message, reveal, noise_only; Node: beacon_reveal, round_key, key_dispute, complaints,
sign_valid_set, release_sum) and sees all public traffic. A is computationally bounded (classical or quantum).

Replica: the deterministic function R(genesis, entries) that every honest party, the driver and the auditor apply
(fedproof/protocol.py: Replica). Lemma 0 below is what makes "the log decides" precise.

Parameters: w = kp + Tn (shares needed to reconstruct a block), quorum q = floor((K + Tn)/2) + 1, decoding needs at
least w + 2*Tn sums, A_min the valid-set floor, H_min honest sites guaranteed inside every opened valid set.

Primitives and assumptions:
- ML-DSA-65 is EUF-CMA secure (FIPS 204). ML-KEM-768 is IND-CCA2 secure (FIPS 203); with AES-256-GCM under a one-time
  key derived from the KEM secret and the associated data, the KEM-DEM is IND-CCA2 (the standard KEM-DEM theorem).
- SHA3-384 and SHAKE-256 are modelled as random oracles (quantum random oracles for the post-quantum statements).
- The Ligero-type argument (fedproof/ligero.py) compiled with salted Merkle commitments and Fiat-Shamir: its round-by-round
  soundness error eps_rbr is the value Params.soundness() reports (proximity gaps for Reed-Solomon codes in the
  unique-decoding regime, per-query and algebraic terms; C6); round-by-round soundness gives soundness of the
  Fiat-Shamir compiled argument in the ROM with error q_H * eps_rbr for q_H hash queries, and in the QROM with error
  O(q_H^2 * eps_rbr), which is why the P256 set keeps about 128 bits against quantum attackers.
- Zero knowledge: with t' >= t masked evaluations the t opened columns are uniformly distributed (rank test,
  tests/test_ligero.py) and the salted Merkle commitments are hiding in the ROM, so the argument is honest-verifier
  zero knowledge and, compiled with Fiat-Shamir, zero knowledge in the ROM.
- Packed Shamir sharing with kp secrets per block and threshold Tn: any Tn share rows are uniform and independent of
  the secrets.

## Lemma 0 (one interpretation of the log)

For every finite bulletin sequence E, every honest party's replica, the driver's replica and audit(genesis, E) reach
the same head, the same round outcomes (valid set, strikes, burns, pointers, model) after every prefix of E.

Proof. Replica.consume is a deterministic function of (state, entry bytes): the head chains the exact wire bytes
(protocol._chain), entries are interpreted only from their decoded wire form, verification results are pure functions
of their inputs (the cache key binds every input), and no step reads a clock, randomness or party-private state.
Every reader receives the same bytes in the same order (bulletin consistency). Induction on the length of E. Entries
that do not encode are refused by the bulletin and never reach a reader. Totality (no exception escapes consume) is
enforced by construction (every handler is wrapped) and was fuzzed with 734 junk entries of 40 shapes. QED.

Consequence: every statement below is about the single outcome R(genesis, E), and the audit recomputes it (C5).

## Lemma 1 (honest parties act only on the log)

An honest party's answer to any request is a function of (its private state, its replica's state). It answers
round_message, round_key, reveal, sign_valid_set and release_sum at most once per round and only when its replica
shows the round state that justifies it (specification.md, "How parties act"). It erases its round key and shares (node) or its
encapsulation seeds (site) exactly when its own replica closes the round.

Proof. By inspection of Site and Node: every public method begins with a check against self.view (the party's own
replica, fixed at setup by the one-shot _attach) and records what it answered; no public method takes a message,
share or view as an argument (tests/test_public_surface.py checks the public surface equals the request list);
erasure is registered as the replica's close listener. QED.

## Theorem 1 (integrity of every counted vector; C1)

If the replica counts a message of site i in round t as proven, then, except with probability at most
q_H * eps_rbr + q_H^2 / 2^384, there exist a data chunk (X, y) and noise bits rho such that
(a) (X, y) opens the data commitment root the log names for chunk c(i, t) and rho opens the noise root the log names
    for site i, which was logged before the round-t beacon;
(b) u_i = Step(W_t, X, y, rho XOR beta_{i,t}), with Step = model.plain_step (exact integers, per-record clipping to
    Chat), W_t the model the log determines, beta_{i,t} = SHAKE(beacon_t, i);
(c) the encrypted share rows and the published digests are consistent with one packed Shamir sharing of u_i.

Proof. The replica verifies the proof against a public input it computes itself (Replica.public_for: weights from
its own model, beta from its own beacon, key_present from its own key set) and the roots it holds; it counts the
message only if the signed context equals its own context (round, site, model digest, beacon, chunk, next noise root,
keys digest), so the statement proved is the statement of round t. By soundness of the compiled argument the
committed witness satisfies the constraint system of fedproof/step.py; the constraint system is the reference semantics
(tests/test_step*.py check bit-identity on all model shapes; tests/test_constraint_coverage.py shows every cell of the witness is
in some constraint except label cells that are range-checked). Binding of the Merkle commitments (collision
resistance of SHA3-384) ties the opened data to the logged roots. The noise root was logged in a previous round or at
registration, hence before beacon_t, and the beacon includes at least Tn + 1 valid chain reveals (at least one honest
node's fresh value), so rho is fixed before beta. Part (c) is a relation inside the same constraint system. QED.

## Lemma 2 (an honest node holds a valid share of every site in the valid set)

Let i be in the valid set V of round t and m an honest node whose round key was accepted. Then, except with
probability q_H * |F|^-G, node m holds the m-th share row of the sharing in Theorem 1(c).

Proof. The ciphertexts are absorbed into the transcript before the challenge gamma (siteproof._transcript), so the
row (s', r') encrypted to m is fixed before gamma. The node accepts its row only if, for each of the G checks,
<gamma_g, s'> + r'_g = D[m]_g, and Theorem 1(c) gives D[m]_g = <gamma_g, s_m> + r_{m,g} for the committed row. If
s' differs from s_m, each equation holds with probability at most 1/|F| over gamma_g; a union bound over hash queries
gives the claim. If the check fails, m posts a signed complaint; the site either reveals a seed, and then anyone
re-derives the ciphertext and reruns the check (protocol.judge), or does not, and in both cases i leaves V
(Replica.judged). So every i that stays in V passed the check of every honest node with a key. QED.

## Theorem 2 (one exact aggregate per round, liars identified; C3)

In every round, (a) at most one valid set is certified; (b) if a result is applied, it equals sum over i in V of u_i,
where V is the replica's valid set and |V| >= A_min; (c) every node that the decoding blames released a wrong sum;
(d) no honest node is ever blamed.

Proof. (a) A certificate counts only with q valid signatures, from non-burned nodes, on the digest of (t, V, head).
Two certified sets need two quorums, which share at least 2q - K >= Tn + 1 nodes, hence an honest node; an honest node
signs at most one digest per round (Lemma 1). (b) An honest node releases once, only after its replica shows a
certificate over the set it signed, the sum of its share rows over V (Lemma 2: these are valid shares). At most Tn
released sums are wrong. Decoding needs at least w + 2Tn sums, so the Reed-Solomon code corrects up to
cap = floor((#sums - w)/2) >= Tn errors. The random combination locates errors uniquely; the full vectors of every
non-base node are then checked against the base. If the base contains a liar (the combination missed it), at least
#sums - w - L + 1 >= cap + 1 honest nodes disagree with the base, the base is discarded and nobody is blamed; with an
honest base the vectors that disagree are exactly the wrong ones. A result is applied only if one base passes, so the
decoded secrets are the sum of the true shares. (c) and (d) follow from the same case split. The probability that
decoding aborts because the combination misses a liar in every attempt is at most (1/|F|)^attempts per liar and
affects liveness only. QED.

## Theorem 3 (privacy of an honest site's data and update; C2, C3)

There is a simulator that, given only the public parameters, the sizes of all messages, the honest sites' committed
roots, and for each round that produced a result the aggregate sum over V of u_i, outputs a view computationally
indistinguishable from the real view of A (relay, at most Tn corrupted nodes per round, any corrupted sites), against
classical and quantum A.

Proof (hybrids).
H0: the real execution.
H1: every honest site's proof is replaced by a simulated proof for the same public statement. Indistinguishable by
zero knowledge of the compiled argument (ROM programming of the Fiat-Shamir challenges, hiding of salted Merkle
commitments and uniform opened columns).
H2: every ciphertext from an honest site to an honest node is replaced by an encryption of zeros. Indistinguishable by
IND-CCA2 of the KEM-DEM: an honest node's round decapsulation key never leaves the node and is erased at close
(Lemma 1), and an encapsulation seed of an honest site is revealed only for a logged complaint by that node (Lemma 1);
an honest node complains against an honest site's message never (its row passes the check), so no seed of an honest
site to an honest node is ever revealed. Decryption queries for other ciphertexts are answered by the reduction
because all other parties' keys are known to it.
H3: digests D[m] of honest nodes are replaced by uniform values. Identical distribution: D[m]_g = <gamma_g, s_m> + r_{m,g}
with the mask r_{m,g} uniform, secret to node m and used once.
H4: the share rows sent to corrupted nodes are replaced by shares of zero. Identical distribution: at most Tn rows of a
packed Shamir sharing with threshold Tn are uniform and independent of the secrets.
H5: the honest nodes' released sums are computed from the logged aggregate alone. In a round, honest nodes release
only over the single certified V (Theorem 2(a)), and by H4 the corrupted nodes' rows of the honest sites are
independent of u; the honest released sums are then a deterministic function of the corrupted rows and of the true
sum over V, which the simulator has. A single site's vector is never opened, because |V| >= A_min is enforced before
certification (the parameter rule A_min >= H_min + corrupted sites >= 2 makes this a sum over several sites, at least
H_min of them honest) and an honest node never releases over any other set.
H6 is the simulator's output. The forward-secrecy part: a node seized after round t closes holds no round-t key or share
(erased) and its long-term signing key gives no information on ciphertexts (round keys are fresh per round), so
seizing changes nothing in H2 to H5 for round t (C3b, with the corruption accounting of Section 0). QED.

## Theorem 4 (record-level differential privacy of the released aggregates; C4)

Fix a record x of an honest site, used in n_uses rounds (chunk_uses counts every proven use, also in rounds that later
abort, so releases the relay sees and then aborts are counted). Each release is the sum over V of u_i, with at least
H_min honest sites in V, each adding noise sum_j sum_k 2^k (rho XOR beta) whose distribution is the sum of J uniform
r-bit integers independent of everything else (rho uniform and secret, fixed before beta, Theorem 1(a)). Replacing x
changes the sum of truncated gradients by at most Delta = 2*Chat/2^sh + sqrt(d) in l2 (exact clipping to Chat is
proved, Theorem 1(b)). Then the sequence of releases is (epsilon, delta)-DP with epsilon = dp.account(cfg, H_min,
n_uses, delta).

Proof. The aggregate noise from H_min honest sites is a sum of H_min*J independent uniform integers per coordinate
(the corrupted sites' noise is ignored, which can only help the adversary less). dp.UniformSum computes its exact
distribution; dp.account bounds the Renyi divergence of order alpha between the noise and its shift by any integer
vector of l2 norm at most Delta, after cutting the tails at total probability delta0 * n_uses (central_cut), with the
maximum over every integer shift up to the bound checked exactly for the chosen order (kappa with stride 1); it
composes over n_uses releases and converts to (epsilon, delta) (rdp_to_dp); tests/test_dp.py checks it against brute force.
Post-processing (the model update) preserves the guarantee. QED.
Remark: the adversary's beacon grinding selects only public bits beta; it cannot bias an honest site's noise, because
rho is uniform, secret and fixed before beta.

## Theorem 5 (accountability; no honest party is blamed)

(a) A site is struck only if the log holds its valid signature on a message whose proof fails, or a seed it signed
that opens a share failing the public check; an honest site is never struck. (b) A node is burned only if the log
holds its signed complaint against a share that the site's revealed seed shows correct, its signed sum that decoding
proves wrong (Theorem 2(c)), or a key validly signed under its name that it disputes (its signing key is exposed);
an honest node is never burned while its signing key is secret. (c) Omissions the relay can cause (a missing reveal,
key, message, seed or noise commitment) cost the party the round and nothing more.

Proof. By inspection of Replica._close and _finalize_keys, together with Theorem 1 (an honest site's proof verifies:
it proves only against the data and noise roots its replica names, with the context its replica computes, so a
counted honest message always verifies), Lemma 2 (an honest site's rows pass every honest node's check) and
EUF-CMA (nobody else can produce the signed evidence). tests/test_log_driven_parties.py and tests/test_public_surface.py
replay an attack on each path. QED.

## Theorem 6 (public verifiability; C5)

The final model is the deterministic function R(genesis, E) of the log; audit(genesis, E) recomputes it, verifying
every proof again without any cache, and raises exactly when a logged result contradicts the recomputed one.

Proof. Lemma 0, plus Replica._check_result, which compares every logged result with the recomputed record of its
round (valid set, aggregate, model digest). QED.

## Liveness (stated, not proved formally)

If the relay delivers every honest party's answers, at most Tn nodes misbehave, at least A_min sites are honest and
online and K >= w + 2*Tn, every round produces a result. This is checked by the honest-liveness runs
(tests/test_honest_liveness.py: several configurations, skipped round numbers, record budgets) and is what the
experiments exercise. A censoring relay can stop progress (specification.md limitations); it cannot change an outcome.

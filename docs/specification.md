# Protocol specification

What the reference implementation does and what it claims. Sites are the plants of the paper; the coordinator is the
relay; the bulletin is the public log.

## Parties and trust
N sites (data owners), K committee nodes, an untrusted coordinator that only relays messages through a public
append-only bulletin, and any number of public auditors. Assumptions: at most Tn nodes are corrupted during any one
round, counted from the posting of that round's keys to their erasure, where a node whose long-term signing key has
ever been exposed counts as corrupted until it is re-registered (corrupted nodes may collude, lie and go offline; an
attacker may seize different nodes in different rounds and read the whole log afterwards); corrupted sites are arbitrary; at least H_min honest sites are in the valid set of every round
that opens an aggregate (A_min must be set so that this holds given the number of corrupted sites); the registry of sites and nodes is fixed and authentic; hash functions are modelled as
random oracles; ML-KEM-768, ML-DSA-65, AES-256-GCM and SHA-3 are secure (so every guarantee, the privacy
guarantee included, holds against computationally bounded adversaries); every party reads the same append-only
bulletin, which the coordinator can delay, censor, reorder or pad with junk, but not rewrite.

## How parties act
Every honest site and node keeps its own Replica (fedproof/protocol.py), a deterministic interpreter of the bulletin. The
driver and the auditor run the same code on the same entries, so all of them reach the same outcome for every round.
A party answers a request from the relay only when its own replica shows that the bulletin justifies it (a round key
only after that round's beacon and before keys are logged; a seed only for a logged, valid complaint by that node; a
sum only after a logged certificate over the exact set the node signed), and at most once. Shares reach a node only
by decrypting the logged ciphertexts. Erasure and every pointer move happen when the party's own replica sees the
round close; no relay call can trigger or prevent them. Entries the rules do not justify (unsigned, signed by the
wrong party, for another round or context, late, duplicated) are ignored by everyone, and hashed into the bulletin
head like any entry. The bulletin stores bytes: each entry is encoded once in the wire format (fedproof/wire.py; anything
that does not encode is refused), every reader gets the decoded form of those bytes, and the head is a hash chain over
them. Message hashing for signatures is canonical: types, lengths, array dtypes and shapes are bound (list and tuple
distinct), so a signature covers the exact message structure. The only public methods of a site are round_message,
reveal and noise_only, and of a node beacon_reveal, round_key, key_dispute, complaints, sign_valid_set and
release_sum, and every request refuses an argument of the wrong type before it records anything; a party's replica and
its committed data are fixed at setup. In-process, the parties' attributes are test instrumentation; the boundary the
threat model relies on is the process boundary of the networked harness, where a party answers exactly these requests,
drops malformed or oversized frames and carries on, and the router tags every response with the connection it came
from, so one party cannot answer for another. Wire decoding is strict: only the canonical encoding of a value decodes
(minimal integer lengths, no duplicate or colliding keys, an element budget), so equal entries always have equal bytes
and the audit, which re-encodes entries, computes the same head as the parties, which chain the bytes they received.

## One round t (fedproof/protocol.py: run_round, continue_round; Replica)
1. Beacon: each node reveals the value of a hash chain it committed to at setup for round t. Round numbers strictly
   increase, by at most MAX_STEP = 4; a reveal is checked by hashing forward to the node's previous one. A node reveals
   for round numbers up to the furthest one it has already revealed in the current gap (a value for t also gives every
   earlier value, so this reveals nothing new), and never further. Reveals are
   unsigned, so a missing or wrong reveal only makes the node sit this round out (no key, no shares). A beacon entry
   with fewer than Tn + 1 good reveals, not counting nodes that closing the open round removes, is ignored. beacon_t = H(t, bulletin head, good reveals); per-site public bits
   beta = SHAKE(beacon_t, site). A beacon for a later round closes an open round as abandoned.
1b. Round keys: each node that revealed generates one fresh ML-KEM-768 key pair for round t and returns the same
   signed key if asked again. A key that is missing, fails the FIPS 203 check or has a bad signature leaves the node
   out of this round only. A node that sees a key logged under its name that carries a valid signature by its own
   signing key but that it did not make (asked for a key or not) posts a signed dispute and is removed for good (its
   signing key is exposed); a junk key in its name is not disputed. Keys become final at the first site entry; their digest enters every
   round context.
2. Each expected site (not excluded, with a registered noise commitment, chunk within max_uses) posts one signed
   message (fedproof/message.py: site_prove) that carries its round context ctx = H(registry, t, site, model digest,
   beacon, chunk, next noise root, keys digest): Merkle roots, one ciphertext per node with a key (empty otherwise),
   digests D (K x G), the root of its next noise commitment, and the zero-knowledge proof. Asked twice, it returns
   the same message; a noise commitment is opened at most once.
3. The proof (fedproof/ligero.py, fedproof/step.py) shows, for the public model of round t: u = floor(U / 2^sh) + noise where
   U is the sum over the b records of the scheduled committed data chunk of the per-record gradient, each scaled so
   that its l2 norm is at most Chat (fedproof/model.py: plain_step is the reference semantics, exact integers; hidden
   layers are ReLU with an upper clamp, the output stage is a one-vs-all smoothed hinge p = clamp(z/4 + 1/2, 0, 1)
   with error p - label, and inputs may be centred by a public offset), and
   noise_i = sum of J uniform r-bit integers whose bits are rho XOR beta, rho being the bits the site committed
   before beacon_t existed. It also shows that the digests are the stated linear functions of the shares, and that
   the re-committed data chunk equals the old one.
4. Only the first site entry with the site's valid signature and the expected ctx counts. A signed proof that fails
   is a strike (more than max_strikes = permanent exclusion). A missing message, or a signed one for another
   context, costs the site the round and its noise commitment (it must post a signed fresh one), but no strike,
   because the relay could have caused it.
5. Each node decrypts its own ciphertext from the counted message and checks it against the digests. A node with an
   accepted key whose check fails posts a signed complaint; the site answers with a signed reveal of the
   encapsulation seed, and anyone re-derives the ciphertext (judge). A revealed bad share is a strike for the site; a
   revealed good share burns the node; a missing reveal removes the site from this round only. Complaints from nodes
   without an accepted key, and complaints or reveals after the certificate, are ignored.
6. If fewer than A_min sites remain valid, the round aborts. Otherwise each honest node signs, once per round, the
   digest of (t, valid set, bulletin head) only if the set equals the one its own replica computes;
   quorum = floor((K+Tn)/2)+1. A certificate counts only if its set equals the replica's set and it carries a quorum.
   Each node then releases once the sum of its shares over that set. Anyone decodes the sums with Reed-Solomon error
   correction (decode_aggregate: at least w + 2*Tn sums; every node's full vector is checked against the decoded
   base; no more nodes are blamed than the code corrects, (sums - w)/2, and if more disagree the base is treated as wrong and a fresh combination
   is tried, so a liar missed by the random combination causes at worst an abort) and applies the public update rule
   (model.apply_update). A logged result is checked against the computed record of its round (a result logged
   before the round closes is checked when it closes).
7. Close (result, abort, failed certificate or decode, or abandonment): every replica applies strikes, burns and
   pointer moves; every node erases its round key and shares, and every site its encapsulation seeds.
8. audit(genesis, entries) runs a fresh replica over the log, verifying every proof again; it raises AssertionError
   only when a logged result contradicts the computation. A strict failure means the log contains a false result
   entry, which the relay can always append: the recomputed model is still the right one (strict=False returns it).
Malformed input never makes a verification entry point raise (well_formed, check_site, site_verify, ligero.verify,
judge, decode_aggregate, Replica.consume, audit). A site never opens a data commitment more than (ell/2 + t')/t times
(zero-knowledge budget), even if its messages are censored.

## Security claims
C1 Integrity/soundness: no site can get a message accepted unless u is exactly the reference-semantics vector for
   its committed chunk, the committed noise bits and the public beta (except with the stated soundness error).
C2 Zero knowledge: the public transcript of an honest site reveals nothing about its data, its update or its noise
   beyond the opened aggregate (shares to at most Tn nodes included).
C3 Aggregation: with at most Tn corrupted nodes and K >= kp + 3*Tn + f, the opened value is the exact sum of the
   valid sites' vectors; lying nodes are identified; a single site's vector is never opened; two different valid
   sets cannot both be certified in one round.
C3b Forward secrecy: an attacker who seizes any set of nodes after round t ends learns nothing about round t from
   them beyond the public log. Seizing a node exposes its long-term signing key, so from then on it counts as
   corrupted in every round until it is re-registered (re-registration is not implemented in the prototype): a relay
   holding that key can post round keys in its name and suppress the node's dispute. The threshold Tn therefore
   bounds, per round, the nodes corrupted in that round plus the nodes whose signing keys were seized earlier.
C4 Privacy accounting (fedproof/dp.py): record-level (epsilon, delta) for a record used n_uses times, with noise from
   H_min honest sites, sensitivity Delta = 2*Chat/2^sh + sqrt(d).
C5 Public verifiability: the final model is a deterministic function of the bulletin; tampering is detected.
C6 Parameter soundness (fedproof/ligero.py: Params.soundness and statement_soundness): round-by-round error from BCIKS20 unique-decoding
   proximity gaps (affine, error n/|F| per repetition), per-query error max(1-(e+1)/n, (e+2k-2)/n) with t distinct
   queries, sigma repetitions of the algebraic tests over the 64-bit Goldilocks field.
Stated limitations (not gaps): corruption bounds above; a site may commit poisoned data (bounded per-record
influence, attributable); model size bounded by proving cost; verifier time linear in the statement; liveness
needs synchrony; epsilon is a chosen parameter; no amplification by subsampling (public schedule); erasure is a
protocol obligation (the implementation drops every reference to the key and the shares; wiping memory and storage
securely is left to a deployment, as in the standard erasure model); erasure needs the round to close, so a relay that stalls an
open round delays it until the next beacon (a deployment adds a local timeout); a relay can censor any entry, which
costs liveness and can keep honest parties out of rounds but cannot change an outcome or blame an honest party; a
relay that also suppresses a node's key dispute can use a stolen signing key (the node then counts as corrupted).
Burns are permanent and the quorum is fixed over all K nodes, so an adversary that corrupts different nodes over
time (each caught) ends certification after K - quorum + 1 burns; a chunk whose opening budget is spent by
censorship cannot be re-committed in the prototype. By padding the log before a beacon the relay can choose the
beacon among as many candidates as it can compute; this selects only public bits, so an honest site's noise stays
uniform (its committed bits are secret and fixed first) and a corrupted site's own selectable bias stays within its
clipping bound.

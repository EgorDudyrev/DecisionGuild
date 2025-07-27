from functools import reduce

from bitarray import bitarray
from bitarray.util import zeros as bazeros, subset as basubset

from tqdm.auto import tqdm

from decisionguild.dual_patterns import DualPattern

def join_closure_pattern(
        premises: list[DualPattern], subpremises: list[bitarray], use_tqdm: bool = False
) -> tuple[list[DualPattern], list[bitarray]]:
    premise_list = premises
    premise_id_map = {premise: id for id, premise in enumerate(premise_list)}
    known_controversies: list[bitarray] = [bazeros(len(premise_list)) for _ in premise_list]

    pbar = tqdm(total=len(premise_list), disable=not use_tqdm)
    i = 0
    while i < len(premise_list):
        premises_to_test = ~subpremises[i] & ~known_controversies[i]
        premises_to_test = premises_to_test[:i]
        k = 0
        while premises_to_test.any():
            j = premises_to_test.find(True)
            premises_to_test[j] = False

            p, q = premise_list[i], premise_list[j]

            join = p | q
            if join.is_controversy:
                known_controversies[i][j] = known_controversies[j][i] = True
                superpremises_i, superpremises_j = [], []
                for id, subs in enumerate(subpremises):
                    if subs[i]: superpremises_i.append(id)
                    if subs[j]: superpremises_j.append(id)
                for super_i in superpremises_i:
                    for super_j in superpremises_j:
                        known_controversies[super_i][super_j] = known_controversies[super_j][super_i] = True
                premises_to_test = premises_to_test & ~known_controversies[i][:i]
                continue

            if join in premise_id_map:
                join_id = premise_id_map[join]
                subpremises[join_id] |= subpremises[i] | subpremises[j]
                continue

            k += 1
            join_id = i+k
            premise_list.insert(join_id, join)
            for premise in premise_id_map:
                if premise_id_map[premise] >= join_id:
                    premise_id_map[premise] += 1
            premise_id_map[join] = join_id
            assert all(premise_list[premise_id_map[premise]] == premise for premise in premise_list)

            for ba in subpremises:
                ba.insert(join_id, ba[i] and ba[j])  # if premise is greater than p and q, then it is greater than join
            subpremises.insert(join_id, subpremises[i] | subpremises[j])
            subpremises[join_id][join_id] = True

            # the following cycle is important. although I do not understand why
            for other_id in subpremises[join_id].search(False, 0, join_id):
                subpremises[join_id][other_id] = premise_list[other_id] <= join

            join_subs = subpremises[join_id][:join_id]
            for ba in known_controversies:
                ba.insert(join_id, (ba[:join_id] & join_subs).any())
            join_controversies = ([known_controversies[other][join_id] for other in range(join_id)]+[False]+
                                  [known_controversies[other][join_id] for other in range(join_id, len(premise_list)-1)])
            known_controversies.insert(join_id, bitarray(join_controversies))
            assert len(known_controversies) == len(premise_list)
            assert all(len(ba)==len(premise_list) for ba in known_controversies)

        i += 1
        pbar.total = len(premise_list)
        pbar.refresh()
        pbar.update(1)

    return premise_list, subpremises


def join_closure_atomic(
        premises: list[DualPattern], subpremises: list[bitarray], use_tqdm: bool = False
) -> tuple[list[DualPattern], list[bitarray]]:
    def activate_index(idx, ba):
        ba_new = bitarray(ba)
        ba_new[idx] = True
        return ba_new

    def find_lefter_controversies(
            premise_list: list[DualPattern], subpremise_relation: list[bitarray]
    ) -> list[bitarray]:
        lefter_controversies: list[bitarray] = []  # bitarrays contains all pairs (i, j) s.t. j<i, (premises[i]|premises[j]).is_controversy

        for i, premise in enumerate(premise_list):
            controversies = bitarray(len(premise_list))
            premises_to_test = ~subpremise_relation[i]
            premises_to_test[i:] = False
            while premises_to_test.any():
                j = premises_to_test.find(True)
                premises_to_test[j] = False
                if (premise | premise_list[j]).is_controversy:
                    controversies[j] = True
                    for j_transitive in premises_to_test.search(True, j, i):
                        if subpremise_relation[j_transitive][j]:
                            controversies[j_transitive] = True
                            premises_to_test[j_transitive] = False
            lefter_controversies.append(controversies)
        return lefter_controversies


    def enumerate_order_ideals(
            premise_list: list[DualPattern], subpremise_relation: list[bitarray], known_controversies: list[bitarray]
    ) -> list[bitarray]:
        pbar = tqdm(total=None, disable=not use_tqdm, desc='Enum. join candidates', unit_scale=True)

        order_ideals = [bazeros(len(premise_list))]
        stack = [activate_index(i, bazeros(len(premise_list))) for i in reversed(range(len(premise_list)))]
        while stack:
            antichain = stack.pop()
            antichain_closed = reduce(bitarray.__or__, (subpremise_relation[i] for i in antichain.search(True)), antichain)
            assert not any((known_controversies[i]&antichain_closed).any() for i in antichain_closed.search(True))

            order_ideals.append(antichain_closed)

            candidates_to_add = ~antichain_closed
            candidates_to_add &= ~reduce(bitarray.__or__, (known_controversies[i] for i in antichain.search(True)), bazeros(len(premise_list)))
            candidates_to_add[antichain.find(True):] = False
            next_antichains = []
            while candidates_to_add.any():
                candidate_i = candidates_to_add.find(True)
                candidates_to_add[candidate_i] = False

                next_antichain = activate_index(candidate_i, antichain)
                next_antichains.append(next_antichain)
            stack.extend(next_antichains[::-1])
            pbar.update(1)
        pbar.close()
        return order_ideals

    def order_bitarrays(bitarrays: list[bitarray]) -> list[bitarray]:
        subsumption: list[bitarray] = []
        for ba_idx, ba in tqdm(enumerate(bitarrays), total=len(bitarrays), disable=not use_tqdm, desc='Order premises', unit_scale=True):
            subsumed = activate_index(ba_idx, bazeros(len(bitarrays)))
            candidates = ~bazeros(len(bitarrays))
            candidates[ba_idx:] = False
            while candidates.any():
                i = candidates.find(True, right=True)
                candidates[i] = False

                if basubset(bitarrays[i], ba):
                    transitives = subsumption[i]
                    subsumed[i] = True
                    subsumed |= transitives
                    candidates &= ~transitives
            subsumption.append(subsumed)
        return subsumption

    def join_premises(idxs: bitarray, premises_: list[DualPattern]):
        return reduce(premises_[0].__class__.join, (premises_[i] for i in idxs.search(True)), premises_[0].get_min_pattern())

    def compute_closed_premises(
            premises_: list[DualPattern], subsumption_order: list[bitarray], order_ideals_: list[bitarray]
    ) -> dict[DualPattern, bitarray]:
        premises_closed = dict()
        for order_ideal_i, order_ideal in tqdm(enumerate(order_ideals_), total=len(order_ideals_),
                                               disable=not use_tqdm, desc='Compute joins', unit_scale=True):
            max_premises = bitarray(order_ideal)
            i = len(max_premises)
            while i > 0 and max_premises[:i].any():
                i = max_premises.find(True, 0, i, right=True)
                max_premises &= ~subsumption_order[i]
                max_premises[i] = True  # reintroduce the value that might got remove during the previous line

            join = join_premises(max_premises, premises_)
            if join in premises_closed:
                premises_closed[join] |= order_ideal
            else:
                premises_closed[join] = order_ideal
        return premises_closed

    lefter_controversies = find_lefter_controversies(premises, subpremises)
    order_ideals = enumerate_order_ideals(premises, subpremises, lefter_controversies)
    closed_premises_dict: dict[DualPattern, bitarray] = compute_closed_premises(premises, subpremises, order_ideals)
    assert not any(p.is_controversy for p in closed_premises_dict)

    premises_closed = sorted(closed_premises_dict, key=lambda p: closed_premises_dict[p].count())
    subsumption_closed = order_bitarrays([closed_premises_dict[p] for p in premises_closed])

    return premises_closed, subsumption_closed

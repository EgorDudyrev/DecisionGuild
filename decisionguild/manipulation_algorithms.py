from bitarray import bitarray
from bitarray.util import zeros as bazeros

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

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> [1] [2] [3] [1,2] [1,3] [2,3] [1,2,3]"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def support_count(itemset, dataset):
    return sum(1 for t in dataset if all(i in t for i in itemset))

data = [
    ['apple', 'bread', 'butter'],
    ['bread', 'jam', 'tea'],
    ['apple', 'bread', 'butter', 'tea'],
    ['carrot', 'bread', 'butter'],
    ['apple', 'bread', 'tea', 'jam'],
    ['apple', 'bread', 'butter', 'jam']
]

all_items = set(chain(*data))
all_combinations = list(powerset(all_items))

min_supp = 0.4
freq_itemsets = {itemset: support for itemset, support in {i: support_count(i, data) / len(data) for i in all_combinations}.items() if support >= min_supp}

def generate_rules(frequent_itemsets, dataset, min_conf=0.6):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for ant in powerset(itemset):
                cons = tuple(set(itemset) - set(ant))
                if cons:
                    ant_supp = support_count(ant, dataset) / len(dataset)
                    rule_supp = support_count(itemset, dataset) / len(dataset)
                    conf = rule_supp / ant_supp
                    if conf >= min_conf:
                        rules.append((ant, cons, rule_supp, conf))
    return rules

assoc_rules = generate_rules(freq_itemsets, data, min_conf=0.7)

for rule in assoc_rules:
    ant, cons, supp, conf = rule
    print(f"Rule: {ant} -> {cons}, Support: {supp:.2f}, Confidence: {conf:.2f}")

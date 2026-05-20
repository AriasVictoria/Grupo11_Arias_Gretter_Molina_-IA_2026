"""
Solución generada íntegramente por gemini (Google) para la Entrega 2.
Problema: diseño del campamento base marciano modelado como CSP.
"""
import itertools

from simpleai.search import (
    CspProblem,
    backtrack,
    MOST_CONSTRAINED_VARIABLE,
    LEAST_CONSTRAINING_VALUE,
)

def build_camp(camp_size, habs, generators, labs, deposits, airlocks, craters):
    rows, cols = camp_size
    craters_set = set(craters)

    
    all_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in craters_set
    ]

    def on_border(r, c):
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def are_adjacent(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    border_cells = [cell for cell in all_cells if on_border(*cell)]
    inner_cells = [cell for cell in all_cells if not on_border(*cell)]

    
    if habs > 0 and len(inner_cells) == 0:
        return None
    if airlocks > 0 and len(border_cells) == 0:
        return None
    if labs > 0 and deposits == 0:
        return None

    module_counts = [
        ("hab", habs),
        ("gen", generators),
        ("lab", labs),
        ("dep", deposits),
        ("air", airlocks),
    ]

    variables = [
        (tipo, i)
        for tipo, count in module_counts
        for i in range(count)
    ]

    if not variables:
        return []

    domain_map = {"air": border_cells, "hab": inner_cells}
    domains = {
        var: list(domain_map.get(var[0], all_cells))
        for var in variables
    }

   
    by_type = {tipo: [] for tipo, _ in module_counts}
    for var in variables:
        by_type[var[0]].append(var)

    hab_vars = by_type["hab"]
    gen_vars = by_type["gen"]
    lab_vars = by_type["lab"]
    dep_vars = by_type["dep"]

    constraints = []

    def no_overlap(variables, values):
        return values[0] != values[1]

    for pair in itertools.combinations(variables, 2):
        constraints.append((pair, no_overlap))

    
    def not_adjacent(variables, values):
        return not are_adjacent(values[0], values[1])

    for gv in gen_vars:
        for hv in hab_vars:
            constraints.append(((gv, hv), not_adjacent))

    for gv1, gv2 in itertools.combinations(gen_vars, 2):
        constraints.append(((gv1, gv2), not_adjacent))

   
    def lab_has_adjacent_deposit(variables, values):
        lab_pos = values[0]
        return any(are_adjacent(lab_pos, dep_pos) for dep_pos in values[1:])

    for lv in lab_vars:
        constraints.append((tuple([lv] + dep_vars), lab_has_adjacent_deposit))

    def make_evacuation_constraint(rows, cols, craters_set):
        def has_evacuation_route(variables, values):
            hab_pos = values[0]
            occupied = set(values[1:])
            r, c = hab_pos
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in occupied and (nr, nc) not in craters_set:
                        return True
            return False
        return has_evacuation_route

    evac = make_evacuation_constraint(rows, cols, craters_set)
    for hv in hab_vars:
        other_vars = [v for v in variables if v != hv]
        constraints.append((tuple([hv] + other_vars), evac))

    problem = CspProblem(variables, domains, constraints)
    solution = backtrack(
        problem,
        variable_heuristic=MOST_CONSTRAINED_VARIABLE,
        value_heuristic=LEAST_CONSTRAINING_VALUE,
        inference=True,
    )

    if solution is None:
        return None

    return [(tipo, r, c) for (tipo, _), (r, c) in solution.items()]

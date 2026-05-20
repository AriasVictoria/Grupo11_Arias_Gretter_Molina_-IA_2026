# coding: utf-8
import itertools
from simpleai.search import CspProblem, backtrack, MOST_CONSTRAINED_VARIABLE, LEAST_CONSTRAINING_VALUE


def build_camp(camp_size, habs, generators, labs, deposits, airlocks, craters):
    filas, columnas = camp_size
    crateres = set(craters)

    # helpers para no repetir logica
    def es_borde(f, c):
        return f == 0 or f == filas - 1 or c == 0 or c == columnas - 1

    def son_adyacentes(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    # separo las celdas disponibles segun donde pueden ir los modulos
    celdas_libres = [(f, c) for f in range(filas) for c in range(columnas) if (f, c) not in crateres]
    celdas_borde = [(f, c) for f, c in celdas_libres if es_borde(f, c)]
    celdas_interior = [(f, c) for f, c in celdas_libres if not es_borde(f, c)]

    # casos imposibles antes de armar el csp
    if habs > 0 and not celdas_interior:
        return None
    if airlocks > 0 and not celdas_borde:
        return None
    if labs > 0 and deposits == 0:
        return None

    # cada modulo es una variable, identificada por tipo e indice
    variables = []
    for i in range(habs):
        variables.append(("hab", i))
    for i in range(generators):
        variables.append(("gen", i))
    for i in range(labs):
        variables.append(("lab", i))
    for i in range(deposits):
        variables.append(("dep", i))
    for i in range(airlocks):
        variables.append(("air", i))

    if not variables:
        return []

    # el dominio de cada variable son las celdas donde puede ir segun su tipo
    # las restricciones unarias las resolvemos aca directamente
    dominios = {}
    for v in variables:
        if v[0] == "air":
            dominios[v] = list(celdas_borde)      # esclusas solo en el borde
        elif v[0] == "hab":
            dominios[v] = list(celdas_interior)   # habitacionales solo adentro
        else:
            dominios[v] = list(celdas_libres)     # el resto en cualquier lado libre

    habs_v = [v for v in variables if v[0] == "hab"]
    gens_v = [v for v in variables if v[0] == "gen"]
    labs_v = [v for v in variables if v[0] == "lab"]
    deps_v = [v for v in variables if v[0] == "dep"]

    restricciones = []

    # no puede haber dos modulos en la misma celda
    def distinto_lugar(vars, vals):
        return vals[0] != vals[1]

    for par in itertools.combinations(variables, 2):
        restricciones.append((par, distinto_lugar))

    # generador no puede estar al lado de un habitacional ni de otro generador
    def no_adyacentes(vars, vals):
        return not son_adyacentes(vals[0], vals[1])

    for g in gens_v:
        for h in habs_v:
            restricciones.append(((g, h), no_adyacentes))

    for g1, g2 in itertools.combinations(gens_v, 2):
        restricciones.append(((g1, g2), no_adyacentes))

    # cada lab tiene que tener al menos un deposito al lado
    def lab_tiene_deposito_cerca(vars, vals):
        lf, lc = vals[0]
        for df, dc in vals[1:]:
            if abs(lf - df) + abs(lc - dc) == 1:
                return True
        return False

    for l in labs_v:
        restricciones.append((tuple([l] + deps_v), lab_tiene_deposito_cerca))

    # cada habitacional necesita al menos una celda libre al lado para evacuar
    def hacer_restriccion_evacuacion(filas, columnas, crateres):
        def tiene_salida(vars, vals):
            f, c = vals[0]
            ocupadas = set(vals[1:])
            for df, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nf, nc = f + df, c + dc
                if 0 <= nf < filas and 0 <= nc < columnas:
                    if (nf, nc) not in ocupadas and (nf, nc) not in crateres:
                        return True
            return False
        return tiene_salida

    fn_evacuacion = hacer_restriccion_evacuacion(filas, columnas, crateres)
    for h in habs_v:
        otros = [v for v in variables if v != h]
        restricciones.append((tuple([h] + otros), fn_evacuacion))

    problema = CspProblem(variables, dominios, restricciones)
    resultado = backtrack(
        problema,
        variable_heuristic=MOST_CONSTRAINED_VARIABLE,
        value_heuristic=LEAST_CONSTRAINING_VALUE,
        inference=True,
    )

    if resultado is None:
        return None

    return [(tipo, f, c) for (tipo, _), (f, c) in resultado.items()]

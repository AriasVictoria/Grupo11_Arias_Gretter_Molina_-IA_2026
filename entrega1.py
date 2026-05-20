from heapq import heappush, heappop
import itertools


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _heuristic(pos, tool, load, ig, sd):
    pending = list(ig) + list(sd)
    if not pending:
        return load

    restantes = list(pending)
    actual = pos
    dist_total = 0
    while restantes:
        d, sig = min((_manhattan(actual, m), m) for m in restantes)
        dist_total += d
        actual = sig
        restantes.remove(sig)

    n = len(pending)
    costo_equipar = 0 if tool is not None else 3
    return dist_total // 2 + n * 2 + (n // 2) * 2 + (n % 2) + costo_equipar


def planear_rover(rover_inicio, bateria_inicial, zonas_sombra,
                  muestras_igneas, muestras_sedimentarias):

    zonas_sombra = set(zonas_sombra)
    ig0 = frozenset(muestras_igneas)
    sd0 = frozenset(muestras_sedimentarias)

    # Estado: (pos, bat, tool, load, ig, sd)
    start = (tuple(rover_inicio), bateria_inicial, None, 0, ig0, sd0)

    h0 = _heuristic(tuple(rover_inicio), None, 0, ig0, sd0)
    counter = itertools.count()
    # heap: (f, g, uid, state, path)
    heap = [(h0, 0, next(counter), start, [])]
    visited = {}

    while heap:
        f, g, _, state, path = heappop(heap)
        pos, bat, tool, load, ig, sd = state

        if state in visited:
            continue
        visited[state] = g

        # goal
        if not ig and not sd and load == 0:
            return path

        r, c = pos

        # generar acciones
        moves = []

        if bat > 1:
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                moves.append(("moverse", (r + dr, c + dc)))

        if bat > 4:
            for dr, dc in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
                moves.append(("sobremarcha", (r + dr, c + dc)))

        if bat > 1:
            if pos in ig and tool != "termico":
                moves.append(("equipar", "termico"))
            if pos in sd and tool != "percusion":
                moves.append(("equipar", "percusion"))

        if bat > 3 and load < 2:
            if pos in ig and tool == "termico":
                moves.append(("recolectar", "ignea"))
            if pos in sd and tool == "percusion":
                moves.append(("recolectar", "sedimentaria"))

        if bat > 1 and load > 0:
            es_ultima = load == 1 and not ig and not sd
            if load == 2 or es_ultima:
                moves.append(("depositar", None))

        if bat < 20 and pos not in zonas_sombra:
            moves.append(("recargar", None))

        for act, target in moves:
            npos, nbat, ntool, nload, nig, nsd = pos, bat, tool, load, ig, sd

            if act == "moverse":
                npos = target
                nbat -= 1
                ng = g + 1
            elif act == "sobremarcha":
                npos = target
                nbat -= 4
                ng = g + 1
            elif act == "equipar":
                ntool = target
                nbat -= 1
                ng = g + 3
            elif act == "recolectar":
                nbat -= 3
                nload += 1
                if target == "ignea":
                    nig = ig - {pos}
                else:
                    nsd = sd - {pos}
                ng = g + 2
            elif act == "depositar":
                nbat -= 1
                ng = g + load
                nload = 0
            elif act == "recargar":
                nbat = min(20, bat + 10)
                ng = g + 4

            ns = (npos, nbat, ntool, nload, nig, nsd)
            if ns not in visited:
                nh = _heuristic(npos, ntool, nload, nig, nsd)
                heappush(heap, (ng + nh, ng, next(counter), ns, path + [(act, target)]))

    return []
from simpleai.search import SearchProblem, astar


class RoverProblem(SearchProblem):

    def __init__(
        self,
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias,
    ):

        self.zonas_sombra = set(zonas_sombra)

        initial_state = (
            rover_inicio,                    # posición
            bateria_inicial,                # batería
            None,                           # taladro equipado
            0,                              # carga
            tuple(muestras_igneas),         # ígneas restantes
            tuple(muestras_sedimentarias),  # sedimentarias restantes
        )
        super().__init__(initial_state)

        # límites del mapa para evitar expansión infinita
        todas = (
            [rover_inicio]
            + list(muestras_igneas)
            + list(muestras_sedimentarias)
            + list(zonas_sombra)
        )

        filas = [f for f, _ in todas]
        columnas = [c for _, c in todas]

        self.min_fila = min(filas) - 2
        self.max_fila = max(filas) + 2
        self.min_col = min(columnas) - 2
        self.max_col = max(columnas) + 2

    def is_goal(self, state):
        _, _, _, carga, igneas, sed = state

        return not igneas and not sed and carga == 0

    def actions(self, state):

        (r, c), bateria, taladro, carga, igneas, sed = state

        acciones = []

        # =========================
        # MOVERSE
        # =========================

        movimientos = []

        pendientes = list(igneas) + list(sed)

        if pendientes:

            objetivo_fila = sum(f for f, _ in pendientes) / len(pendientes)
            objetivo_col = sum(c2 for _, c2 in pendientes) / len(pendientes)

            posibles = [
                (r + 1, c),
                (r - 1, c),
                (r, c + 1),
                (r, c - 1),
            ]

            for nr, nc in posibles:

                distancia_actual = (
                    abs(r - objetivo_fila)
                    + abs(c - objetivo_col)
                )

                nueva_distancia = (
                    abs(nr - objetivo_fila)
                    + abs(nc - objetivo_col)
                )

                if nueva_distancia <= distancia_actual:
                    movimientos.append((nr, nc))

        else:

            movimientos = [
                (r + 1, c),
                (r - 1, c),
                (r, c + 1),
                (r, c - 1),
            ]

        for destino in movimientos:

            nr, nc = destino

            if (
                self.min_fila <= nr <= self.max_fila
                and self.min_col <= nc <= self.max_col
                and bateria > 1
            ):
                acciones.append(("moverse", destino))

        # =========================
        # SOBREMARCHA
        # =========================

        movimientos_overdrive = []

        posibles_overdrive = [
            (r + 2, c),
            (r - 2, c),
            (r, c + 2),
            (r, c - 2),
        ]

        for nr, nc in posibles_overdrive:

            if pendientes:

                distancia_actual = (
                    abs(r - objetivo_fila)
                    + abs(c - objetivo_col)
                )

                nueva_distancia = (
                    abs(nr - objetivo_fila)
                    + abs(nc - objetivo_col)
                )

                if nueva_distancia <= distancia_actual:
                    movimientos_overdrive.append((nr, nc))

            else:
                movimientos_overdrive.append((nr, nc))

        for destino in movimientos_overdrive:

            nr, nc = destino

            if (
                self.min_fila <= nr <= self.max_fila
                and self.min_col <= nc <= self.max_col
                and bateria > 4
            ):
                acciones.append(("sobremarcha", destino))

        # =========================
        # EQUIPAR TALADRO
        # =========================

        if taladro != "termico" and bateria > 1:
            acciones.append(("equipar", "termico"))

        if taladro != "percusion" and bateria > 1:
            acciones.append(("equipar", "percusion"))

        # =========================
        # RECOLECTAR
        # =========================

        if carga < 2:

            if (
                (r, c) in igneas
                and taladro == "termico"
                and bateria > 3
            ):
                acciones.append(("recolectar", "ignea"))

            if (
                (r, c) in sed
                and taladro == "percusion"
                and bateria > 3
            ):
                acciones.append(("recolectar", "sedimentaria"))

        # =========================
        # DEPOSITAR
        # =========================

        if carga > 0 and bateria > 1:

            total_restantes = len(igneas) + len(sed)

            if carga == 2 or (carga == 1 and total_restantes == 0):
                acciones.append(("depositar", None))

        # =========================
        # RECARGAR
        # =========================

        if (
            (r, c) not in self.zonas_sombra
            and bateria < 20
        ):
            acciones.append(("recargar", None))

        return acciones

    def result(self, state, action):

        (r, c), bateria, taladro, carga, igneas, sed = state

        tipo, target = action

        nueva_pos = (r, c)
        nueva_bateria = bateria
        nuevo_taladro = taladro
        nueva_carga = carga

        nuevas_igneas = list(igneas)
        nuevas_sed = list(sed)

        # =========================
        # MOVERSE
        # =========================

        if tipo == "moverse":

            nueva_pos = target
            nueva_bateria -= 1

        # =========================
        # SOBREMARCHA
        # =========================

        elif tipo == "sobremarcha":

            nueva_pos = target
            nueva_bateria -= 4

        # =========================
        # EQUIPAR
        # =========================

        elif tipo == "equipar":

            nuevo_taladro = target
            nueva_bateria -= 1

        # =========================
        # RECOLECTAR
        # =========================

        elif tipo == "recolectar":

            nueva_bateria -= 3
            nueva_carga += 1

            if target == "ignea":
                nuevas_igneas.remove((r, c))

            elif target == "sedimentaria":
                nuevas_sed.remove((r, c))

        # =========================
        # DEPOSITAR
        # =========================

        elif tipo == "depositar":

            nueva_bateria -= 1
            nueva_carga = 0

        # =========================
        # RECARGAR
        # =========================

        elif tipo == "recargar":

            nueva_bateria = min(20, bateria + 10)

        return (
            nueva_pos,
            nueva_bateria,
            nuevo_taladro,
            nueva_carga,
            tuple(nuevas_igneas),
            tuple(nuevas_sed),
        )

    def cost(self, state, action, state2):

        tipo, _ = action

        if tipo == "moverse":
            return 1

        if tipo == "sobremarcha":
            return 1

        if tipo == "equipar":
            return 3

        if tipo == "recolectar":
            return 2

        if tipo == "depositar":

            _, _, _, carga, _, _ = state
            return carga

        if tipo == "recargar":
            return 4

    def heuristic(self, state):

        (r, c), _, _, carga, igneas, sed = state

        pendientes = list(igneas) + list(sed)

        if not pendientes:
            return 0

        distancias = [
            abs(r - pr) + abs(c - pc)
            for (pr, pc) in pendientes
        ]

        # distancia mínima a una muestra
        minima = min(distancias)

        # cada muestra todavía requiere:
        # recolectar = 2 mins
        # depositar = al menos 1 min cada 2 muestras
        costo_muestras = len(pendientes) * 2

        if carga > 0:
            costo_muestras += 1

        return minima + costo_muestras
def planear_rover(
    rover_inicio,
    bateria_inicial,
    zonas_sombra,
    muestras_igneas,
    muestras_sedimentarias,
):

    problema = RoverProblem(
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias,
    )

    resultado = astar(problema)

    if resultado is None:
        return []

    acciones = []

    nodo = resultado

    while nodo.parent is not None:

        acciones.append(nodo.action)
        nodo = nodo.parent

    acciones.reverse()

    return acciones
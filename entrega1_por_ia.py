from simpleai.search import SearchProblem, astar

class RoverProblem(SearchProblem):
    """
    Formulación del problema de navegación del rover Ares-1.
    Estado: (posicion, bateria, taladro_activo, carga, muestras_igneas, muestras_sedimentarias)
    - posicion: tupla (fila, columna)
    - bateria: entero 1-20
    - taladro_activo: None | "termico" | "percusion"
    - carga: entero 0-2
    - muestras_igneas: frozenset de coordenadas pendientes
    - muestras_sedimentarias: frozenset de coordenadas pendientes
    """

    def __init__(self, rover_inicio, bateria_inicial, zonas_sombra,
                 muestras_igneas, muestras_sedimentarias):
        self.zonas_sombra = frozenset(map(tuple, zonas_sombra))

        estado_inicial = (
            tuple(rover_inicio),
            bateria_inicial,
            None,
            0,
            frozenset(map(tuple, muestras_igneas)),
            frozenset(map(tuple, muestras_sedimentarias)),
        )
        super().__init__(estado_inicial)

    def is_goal(self, state):
        _, _, _, carga, igneas, sedimentarias = state
        return not igneas and not sedimentarias and carga == 0

    def actions(self, state):
        pos, bat, taladro, carga, igneas, sedimentarias = state
        r, c = pos
        acciones = []

        # Moverse a celda adyacente (necesita bat > 1 para no llegar a 0)
        if bat > 1:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                acciones.append(("moverse", (r + dr, c + dc)))

        # Sobremarcha: 2 celdas en línea recta (necesita bat > 4)
        if bat > 4:
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                acciones.append(("sobremarcha", (r + dr, c + dc)))

        # Equipar taladro (necesita bat > 1)
        if bat > 1:
            if pos in igneas and taladro != "termico":
                acciones.append(("equipar", "termico"))
            if pos in sedimentarias and taladro != "percusion":
                acciones.append(("equipar", "percusion"))

        # Recolectar muestra (necesita bat > 3 y espacio)
        if bat > 3 and carga < 2:
            if pos in igneas and taladro == "termico":
                acciones.append(("recolectar", "ignea"))
            if pos in sedimentarias and taladro == "percusion":
                acciones.append(("recolectar", "sedimentaria"))

        # Depositar: solo con 2 muestras, o con 1 si es la última
        if bat > 1 and carga > 0:
            es_ultima = carga == 1 and not igneas and not sedimentarias
            if carga == 2 or es_ultima:
                acciones.append(("depositar", None))

        # Recargar (solo fuera de zonas de sombra y sin batería llena)
        if bat < 20 and pos not in self.zonas_sombra:
            acciones.append(("recargar", None))

        return acciones

    def result(self, state, action):
        pos, bat, taladro, carga, igneas, sedimentarias = state
        tipo, parametro = action

        if tipo == "moverse":
            pos = tuple(parametro)
            bat -= 1
        elif tipo == "sobremarcha":
            pos = tuple(parametro)
            bat -= 4
        elif tipo == "equipar":
            taladro = parametro
            bat -= 1
        elif tipo == "recolectar":
            bat -= 3
            carga += 1
            if parametro == "ignea":
                igneas = igneas - {pos}
            else:
                sedimentarias = sedimentarias - {pos}
        elif tipo == "depositar":
            bat -= 1
            carga = 0
        elif tipo == "recargar":
            bat = min(20, bat + 10)

        return (pos, bat, taladro, carga, igneas, sedimentarias)

    def cost(self, state, action, state2):
        tipo, _ = action
        carga = state[3]
        costos = {
            "moverse": 1,
            "sobremarcha": 1,
            "equipar": 3,
            "recolectar": 2,
            "depositar": carga,  # 1 minuto por muestra depositada
            "recargar": 4,
        }
        return costos.get(tipo, 1)

    def heuristic(self, state):
        pos, _, taladro, carga, igneas, sedimentarias = state
        pendientes = list(igneas) + list(sedimentarias)

        if not pendientes:
            return carga  # solo falta depositar si hay carga

        # Estimación del recorrido total por las muestras (greedy nearest neighbor)
        restantes = list(pendientes)
        actual = pos
        dist_total = 0
        while restantes:
            d, siguiente = min(
                (abs(actual[0] - x) + abs(actual[1] - y), (x, y))
                for x, y in restantes
            )
            dist_total += d
            actual = siguiente
            restantes.remove(siguiente)

        n = len(pendientes)

        # Costo mínimo de viaje usando sobremarcha (2 celdas por minuto)
        costo_viaje = dist_total // 2

        # Costo mínimo de operaciones:
        # - recolectar: 2 minutos por muestra
        # - depositar: de a pares (costo 2) + última suelta si n impar (costo 1)
        # - equipar: al menos una vez si aún no tiene taladro
        costo_recolectar = n * 2
        costo_depositar = (n // 2) * 2 + (n % 2)
        costo_equipar = 0 if taladro is not None else 3

        return costo_viaje + costo_recolectar + costo_depositar + costo_equipar


def planear_rover(rover_inicio, bateria_inicial, zonas_sombra,
                  muestras_igneas, muestras_sedimentarias):

    problema = RoverProblem(
        rover_inicio,
        bateria_inicial,
        zonas_sombra,
        muestras_igneas,
        muestras_sedimentarias,
    )

    resultado = astar(problema, graph_search=True)

    if resultado is None:
        return []

    return [accion for accion, _ in resultado.path() if accion is not None]
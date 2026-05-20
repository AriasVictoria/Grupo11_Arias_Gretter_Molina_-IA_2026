# Conclusiones Entrega 1

Para resolver el problema consultamos a Claude (claude.ai) y usamos A* con una formulación propia del estado: posición del rover, batería, herramienta activa, carga actual, y conjuntos de muestras pendientes. La IA llegó a una estructura similar, pero las diferencias aparecieron en los detalles: la batería nunca puede llegar a cero (tiene que quedar al menos 1 después de cada acción), y depositar solo es válido con 2 muestras o si es la última. La solución de Claude tuvo esos mismos bugs y los fuimos corrigiendo juntos.

La heurística también fue un problema. Claude propuso la distancia manhattan a la muestra más cercana, que subestima demasiado y obliga a A* a explorar demasiados estados. Terminamos usando un recorrido greedy por todas las muestras pendientes dividido por 2, lo que mejoró la performance sin romper la admisibilidad.

El problema más grave fue el overhead de SimpleAI: en el caso g2 nunca terminaba dentro del tiempo límite. Lo reemplazamos por nuestra propia implementación de A* con heapq, lo que redujo el tiempo de "no termina" a 0.3 segundos. Claude fue útil como punto de partida, pero la lección más importante fue que elegir bien la librería tiene tanto impacto como el algoritmo en sí.




















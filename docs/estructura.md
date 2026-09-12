# Estructura y convenciones

Este proyecto separa los datos fuente del trabajo de limpieza y de sus
resultados para conservar trazabilidad.

| Ruta | Uso |
| --- | --- |
| `data/raw/` | Datos tal como fueron recibidos. No editar ni sobrescribir. |
| `data/processed/` | Conjuntos generados después de aplicar la limpieza. |
| `notebooks/` | Exploración, diagnóstico y ejecución explicable del proceso. |
| `src/` | Funciones de carga, validación y transformación cuando dejen de ser experimentales. |
| `tests/` | Pruebas de las funciones de `src/` y de las reglas de calidad. |
| `docs/` | Rúbrica, diccionario de datos, decisiones y resultados. |

## Convenciones

- El notebook debe leer desde `../data/raw/` y escribir solo en `../data/processed/`.
- Los artefactos generados deben tener nombres descriptivos y no reemplazar el CSV original.
- Cada decisión de limpieza debe indicar la columna afectada, la regla aplicada y el motivo.
- Cuando una transformación se use más de una vez, muévela a `src/` y añade una prueba en `tests/`.

La rúbrica entregada se conserva como [rubrica.pdf](rubrica.pdf).

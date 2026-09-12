# Diccionario inicial de datos

Fuente: `data/raw/unclean_smartwatch_health_data.csv`.

| Columna | Descripción esperada | Revisión de calidad inicial |
| --- | --- | --- |
| `User ID` | Identificador de usuario. | Hay valores ausentes; comprobar formato, unicidad y duplicados. |
| `Heart Rate (BPM)` | Frecuencia cardíaca en latidos por minuto. | Convertir a numérico y revisar valores fuera de rango. |
| `Blood Oxygen Level (%)` | Saturación de oxígeno como porcentaje. | Convertir a numérico, buscar nulos y validar el rango porcentual. |
| `Step Count` | Cantidad de pasos. | Convertir a numérico; no debería ser negativa. |
| `Sleep Duration (hours)` | Duración del sueño en horas. | Hay valores no numéricos como `ERROR` y valores ausentes. |
| `Activity Level` | Nivel de actividad. | Normalizar variantes como `Actve`, `Seddentary`, espacios y guiones bajos. |
| `Stress Level` | Nivel de estrés. | Convertir a numérico; detectar faltantes y textos como `Very High`. |

Estas son observaciones de la fuente sin limpiar; las reglas finales deben quedar
registradas junto con el análisis reproducible.

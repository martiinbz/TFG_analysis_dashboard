# Dashboard Streamlit del análisis

App interactiva para explorar los JSON de `RESULTADOS_ANALISIS/FINAL_DATASET/results`.

## Ejecutar

```powershell
cd C:\Users\marti\OneDrive\Escritorio\TFG\ANALISIS\streamlit
streamlit run app.py
```

La app carga los JSON directamente, excluye `evidence_payload` y genera:

- resumen del corpus y de las features;
- gráficas comparativas generales;
- explorador configurable por ejes, tipo de gráfico, décadas, géneros y tonos;
- ficha visual de cada película;
- tabla normalizada descargable como CSV.

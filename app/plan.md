# Plan de Implementación: Interfaz Web de Visualización y Comparación de Modelos

El objetivo es crear una interfaz web simple, con una estética moderna (tipo Hugging Face), que permita a los usuarios subir un video, procesarlo y comparar tres resultados diferentes: el video original, el video con keypoints y el video con el análisis de comportamiento (behavior).

## User Review Required

> [!IMPORTANT]  
> Por favor, revisa este plan. Para el desarrollo, estoy proponiendo usar **HTML, CSS (Vanilla) y JavaScript puro** para mantener la carpeta `app` ligera y fácil de integrar con tu código existente (presumiblemente Python). Si prefieres usar un framework como Vite o React, házmelo saber.

## Open Questions

> [!WARNING]  
> 1. **Procesamiento del Video:** Por ahora, crearé la interfaz visual y la lógica del reproductor. ¿Tienes ya un servidor o API (ej. Flask, FastAPI o Gradio) que reciba el video subido y devuelva los tres videos procesados, o quieres que construya un servidor de prueba básico en Python para conectarlo?
> 2. **Sincronización:** Para que la comparación frame a frame sea instantánea, mi propuesta técnica es tener los 3 videos cargados en el navegador y sincronizados en el mismo segundo, mostrando solo el seleccionado al hacer clic en los botones. ¿Estás de acuerdo con este enfoque?

## Proposed Changes

### Interfaz Web (Carpeta `app`)

Se crearán los siguientes archivos estáticos:

#### [NEW] [index.html](file:///workspace/pig-monitoring/app/index.html)
Estructura de la página que incluye:
- Cabecera con título.
- Zona de "Drag and Drop" para subir el video, o hacer clic para abrir el explorador de archivos. Mostrará el nombre del archivo seleccionado o un placeholder.
- Contenedor principal inferior dividido en:
  - Panel izquierdo: 3 botones de selección (`Video original`, `Video con keypoints`, `Behavior`).
  - Panel derecho: Reproductor de video.

#### [NEW] [style.css](file:///workspace/pig-monitoring/app/style.css)
Diseño moderno y premium (inspirado en Hugging Face Spaces):
- Paleta de colores cuidada (tonos oscuros o neutros con acentos vibrantes).
- Diseño de tarjetas (glassmorphism o bordes suaves) y sombras suaves.
- Efectos hover en los botones y en la zona de upload.
- Micro-animaciones para una experiencia de usuario fluida.

#### [NEW] [script.js](file:///workspace/pig-monitoring/app/script.js)
Lógica de la aplicación interactiva:
- Manejo de eventos de arrastrar y soltar (drag & drop) y selección de archivo.
- Lógica para los tres botones: al hacer clic, cambiará la vista al video correspondiente.
- Lógica de Sincronización: Al cambiar entre "Original", "Keypoints" y "Behavior", el reproductor mantendrá el tiempo exacto (`currentTime`) y el estado de pausa/reproducción, permitiendo analizar un frame específico en los tres modelos de forma impecable.

## Verification Plan

### Manual Verification
- Abrir la interfaz y probar el sistema de Drag and Drop con un archivo de video.
- Validar que la interfaz responde visualmente y muestra el nombre del video cargado.
- Validar que los tres botones alternan correctamente la vista.
- Comprobar que, al pausar el video en un frame específico y hacer clic en otro botón, el nuevo video se muestra exactamente en el mismo frame para poder compararlo.

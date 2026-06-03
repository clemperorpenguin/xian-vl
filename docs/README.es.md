# Xian-MAGE — Asistente de Visión y Lenguaje en Tiempo Real para Entornos de Juego 🧙‍♂️🎮🤖

Xian-MAGE es un HUD y asistente de juegos de escritorio persistente y en tiempo real para Linux (Wayland). Impulsado por el backend de **[Lemonade Server](https://lemonade-server.ai/)**, superpone OCR en tiempo real, traducción, visualizadores de clics con anclaje visual y chat conversacional interactivo directamente sobre los entornos de juego activos.

Dado que la inferencia se orquesta a través de Lemonade, **MAGE admite la ejecución acelerada por Vulkan, funcionando sin problemas en GPU AMD Radeon™ y otros aceleradores.**

Véalo en acción en YouTube: https://www.youtube.com/watch?v=Izu_8pql7cE

<img width="400" height="340" alt="mage" src="https://github.com/user-attachments/assets/bb51b2c6-378f-4a3e-b25d-05ad284e374b" />

---

## 🌟 Características Principales (Totalmente Operativas)

- **Aceleración Vulkan / GPU AMD**: Ejecución de modelos de visión y lenguaje de baja latencia impulsada por la aceleración de hardware de la GPU backend.
- **Superposición de Escritorio que Permite Hacer Clic a Través**: Ventanas superpuestas transparentes basadas en PyQt6 que muestran texto traducido directamente sobre los HUD y diálogos del juego, permaneciendo completamente invisibles a las entradas del ratón.
- **Teclas de Acceso Rápido Globales de Wayland y OSD de Comandos**: Active la traducción, las configuraciones del OSD y las barras laterales de manera fluida utilizando teclas líder personalizables en todo el sistema.
- **Modo Diálogo (Auto-jugar VNs / RPGs de Historia)**: Fije una región de la pantalla, traduzca automáticamente y avance/actualice las traducciones en línea con un simple clic del ratón.
- **Resaltado de Objetivos de Anclaje Visual**: Pregunte al asistente *"¿dónde hago clic?"* o *"¿dónde está la salida?"* y observe cómo resalta las coordenadas físicas exactas en su pantalla.
- **Modo Cinemático (Traducción de Voz Contextual)**: Acopla perfectamente el análisis de visión de captura de pantalla con la traducción de reproducción de audio.
- **Diccionario Local CC-CEDICT**: Pase el cursor instantáneamente sobre cualquier burbuja de traducción y presione `Alt` para obtener un desglose de análisis local seguro para subprocesos de caracteres chinos, pinyin y definiciones.

---

## 🛠️ Primeros Pasos (Cliente MAGE)

### Requisitos

- **Linux con Wayland** (también compatible con X11; las combinaciones de teclas globales requieren entradas `evdev`)
- **Permisos de Usuario**: Su usuario debe estar en el grupo `input` para la captura de teclas de acceso rápido globales (`sudo usermod -aG input $USER` y cierre sesión/inicie sesión)
- **Lemonade Server**: Una instancia de Lemonade Server en ejecución (accesible en `http://localhost:13305` por defecto)

### Configuración Rápida (Linux)

Clone el repositorio y ejecute el script de arranque — instala [`uv`](https://docs.astral.sh/uv/), sincroniza todas las dependencias e inicia MAGE automáticamente:

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
./mage.sh
```

Para agregar MAGE al menú de aplicaciones de su escritorio:

```bash
./mage.sh --install
```

Para agregar MAGE a su menú **y** instalar automáticamente las dependencias del sistema, compilar el servidor Lemonade integrable desde el código fuente y descargar el modelo de visión y lenguaje predeterminado:

```bash
./mage.sh --install --build
```

Para eliminar la entrada y el icono del escritorio:

```bash
./mage.sh --uninstall
```

### Versiones Precompiladas (Windows y macOS)

Si está ejecutando Windows o macOS, o prefiere no compilar desde el código fuente, descargue los paquetes precompilados desde la página de [GitHub Releases](https://github.com/clemperorpenguin/xian-vl/releases).

Las versiones vienen en dos variantes:
- **Lite**: Versión ligera e independiente. Requiere conectarse a un Lemonade Server externo en ejecución.
- **Full**: Incluye el servidor integrado `lemond`, que se inicia y detiene automáticamente cuando se ejecuta la aplicación.

#### Windows
1. Descargue `mage-client-Windows-x86-64-lite.zip` o `mage-client-Windows-x86-64-full.zip`.
2. Extraiga el archivo.
3. Haga doble clic en `mage-client.exe` para ejecutarlo.

#### macOS
1. Descargue `mage-client-MacOS-ARM64-lite.dmg` o `mage-client-MacOS-ARM64-full.dmg` (o los equivalentes en ZIP).
2. Haga doble clic en el DMG y arrastre `mage-client.app` a su directorio de **Aplicaciones**.
3. Abra y ejecute la aplicación.
   > [!NOTE]
   > Dado que la aplicación no está notariada/firmada por Apple, Gatekeeper la bloqueará con una advertencia que dice que la aplicación está "dañada y no se puede abrir". Puede solucionar esto fácilmente ejecutando el siguiente comando en su terminal:
   > ```bash
   > xattr -cr /Applications/mage-client.app
   > ```

### Configuración Manual (Todas las Plataformas)

Si prefiere gestionar el entorno usted mismo:

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
uv sync --all-packages
uv run --package mage-client mage
```

### Controles
- **Abrir Menú de Acción / OSD** — Doble toque de `Shift` (Tecla Líder Predeterminada)
- **Activar Captura de Pantalla** — Menú de Acción `C` (seleccione la región de la pantalla, luego Traducir / Diálogo / Chat)
- **Alternar Barra Lateral de Chat** — Menú de Acción `A`
- **Traducir para Chat (Entrada)** — Menú de Acción `T`
- **Panel de Configuración** — Menú de Acción `S`

---

## 📁 Arquitectura y Proyectos Satélite

El monorepositorio contiene el cliente MAGE principal listo para producción, así como andamios experimentales de proyectos complementarios:

```
├── apps/
│   ├── mage-client/      # 🧙‍♂️ La aplicación principal de HUD de juegos verificada y basada en PyQt6
│   ├── nate/             # 📱 Lector OCR y diccionario compañero para Android (Experimental)
│   ├── masha-extension/  # 🌐 Traductor de selección de extensión de navegador (Experimental)
│   ├── lore-client/      # 📜 CLI constructor de wiki de conocimiento RAG (Experimental)
│   └── luduan-client/    # 🦤 CLI de traducción de EPUB y audiolibros (Experimental)
└── packages/
    ├── xian-vl/          # ⚙️ Motor de orquestación LLM/ASR principal y gestores de contexto
    └── shared-types/     # 📦 Modelos canónicos, constantes y tipos compartidos
```

> [!WARNING]
> **Limitación de ASR / Audio**: El envío de cargas de flujos de audio en vivo a Lemonade está actualmente roto debido a limitaciones del backend del servidor. En consecuencia, las funciones de traducción de voz en vivo (como el **Modo Incursión**) están desactivadas. Todas las funciones de OCR visual, traducción de texto, diccionario y anclaje de chat son totalmente funcionales.

---

## 📜 Licencia

Este proyecto está licenciado bajo la Licencia Pública General de GNU v3.0. Consulte el archivo [LICENSE](LICENSE) para más detalles.

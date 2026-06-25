# NATE (Neural Analysis & Translation Engine) 📱

NATE is a minimal Android companion to the `xian-vl` ecosystem: **take a photo,
and it overlays the translated text back onto the image** — Google-Translate-camera
style — using a local **[Lemonade Server](https://lemonade-server.ai/)** for both
OCR and translation. It is free and fully local to your own hardware: **no host
models, no cloud APIs, no paid services**.

## What it does

1. **Take or pick a photo** containing text.
2. NATE sends it to your Lemonade node and gets back, in one call, every text
   region with a bounding box, the original text, and the translation.
3. It **inpaints** the result: each source region is covered with its sampled
   background colour and the translation is drawn fitted in place.

## Lemonade-only by design (custom omni-router)

NATE performs **client-side model discovery** rather than trusting the server's
Omni bundle. `NateOmniRouter` queries `GET /v1/models?show_all=true` and selects
a **`vision`-labelled** model (e.g. `Gemma-4-31B-it-GGUF`) for OCR + translation.
This matters: on some nodes the "Lite" Omni has no vision LLM component and
*refuses* image input, so NATE choosing the vision model itself is what makes the
camera path reliable. It is the Kotlin counterpart of the core
[`OmniModelRouter`](../../packages/xian-vl/src/xian/omni_router.py); see the
ecosystem [architecture doc](../../docs/AI_ARCHITECTURE.md).

All inference goes through the standard OpenAI-compatible endpoints:

| Endpoint | Use |
| --- | --- |
| `GET /v1/models?show_all=true` | discover the vision model |
| `POST /v1/chat/completions` | OCR + translation in a single multimodal call |

The model returns a JSON array of `{"box":[x1,y1,x2,y2],"original","translated"}`
with boxes normalized 0–1000 over the image; NATE strips code fences / reasoning
output and maps boxes to pixels for the overlay.

## Configuration

Open the app and set the **Lemonade node URL** (default `http://192.168.0.183:13305`
— edit for your own host) and the **target language**. Settings persist locally.
NATE talks to the node over plain HTTP on your LAN, so cleartext is permitted in
the network security config.

## Build & run

> Requires Android Studio + SDK (no Gradle wrapper is checked in; open the
> project in Android Studio to provision one).

1. Open `apps/nate` in Android Studio and let Gradle sync.
2. Deploy to a device or emulator on the **same network as the Lemonade node**.
3. Ensure the node serves a `vision`-labelled model.

## Tech stack

- **Kotlin + Jetpack Compose** UI (single screen).
- **Retrofit + OkHttp + Gson** for the Lemonade API.
- **`android.graphics` Canvas** for the inpaint overlay.
- No Chaquopy, PaddleOCR, MLKit, or Room — those host-model dependencies were
  removed; the device does no model inference.

## Status & limitations

- **Verified**: the OCR + bounding-box + translation contract is confirmed
  against a live Lemonade node (boxes accurate to a few pixels; translations
  correct). The Android UI has **not** been built/run in CI — it needs an Android
  Studio build pass.
- **Latency**: a large vision model takes roughly **20–40 s** per photo (accurate
  boxes require the model's reasoning pass). This is a snap-then-wait flow, not
  real-time.
- **Inpaint** is a flat background-fill + refitted text (robust, on-device), not
  generative inpainting; busy photographic backgrounds will show a solid patch.

## License

GNU General Public License v3.0

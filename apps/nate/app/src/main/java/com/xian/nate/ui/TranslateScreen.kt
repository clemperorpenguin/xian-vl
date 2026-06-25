/*
 * NATE — Lemonade-powered camera translator (Android).
 * Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)
 */

package com.xian.nate.ui

import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.xian.nate.api.NateOmniRouter
import com.xian.nate.api.ServiceFactory
import com.xian.nate.ocr.VisionTranslator
import com.xian.nate.render.Inpainter
import com.xian.nate.util.Images
import com.xian.nate.util.Settings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranslateScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val settings = remember { Settings(context) }

    var nodeUrl by remember { mutableStateOf(settings.nodeUrl) }
    var targetLang by remember { mutableStateOf(settings.targetLang) }
    var original by remember { mutableStateOf<Bitmap?>(null) }
    var result by remember { mutableStateOf<Bitmap?>(null) }
    var showResult by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var pendingCameraUri by remember { mutableStateOf<Uri?>(null) }

    fun load(uri: Uri?) {
        uri ?: return
        original = Images.decode(context, uri)
        result = null
        showResult = false
        status = if (original == null) "Could not read that image." else null
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { uri -> load(uri) }

    val cameraLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture(),
    ) { ok -> if (ok) load(pendingCameraUri) }

    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            val uri = Images.newCameraUri(context)
            pendingCameraUri = uri
            cameraLauncher.launch(uri)
        } else {
            status = "Camera permission denied."
        }
    }

    fun translate() {
        val bmp = original ?: return
        settings.nodeUrl = nodeUrl
        settings.targetLang = targetLang
        busy = true
        status = "Recognizing & translating… (a vision model can take ~30s)"
        scope.launch {
            try {
                val rendered = withContext(Dispatchers.IO) {
                    val service = ServiceFactory.create(nodeUrl)
                    val router = NateOmniRouter(service).apply { discover() }
                    val regions = VisionTranslator(service, router).translate(bmp, targetLang)
                    if (regions.isEmpty()) null else Inpainter.render(bmp, regions) to regions.size
                }
                if (rendered == null) {
                    status = "No text found in the image."
                } else {
                    result = rendered.first
                    showResult = true
                    status = "Translated ${rendered.second} region(s)."
                }
            } catch (e: Exception) {
                status = "Failed: ${e.message ?: e.javaClass.simpleName}"
            } finally {
                busy = false
            }
        }
    }

    Scaffold(topBar = { TopAppBar(title = { Text("NATE — Camera Translate") }) }) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedTextField(
                value = nodeUrl,
                onValueChange = { nodeUrl = it },
                label = { Text("Lemonade node URL") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )

            LanguageDropdown(targetLang) { targetLang = it }

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(
                    onClick = { cameraPermission.launch(android.Manifest.permission.CAMERA) },
                    enabled = !busy,
                    modifier = Modifier.weight(1f),
                ) { Text("Take photo") }
                OutlinedButton(
                    onClick = { galleryLauncher.launch("image/*") },
                    enabled = !busy,
                    modifier = Modifier.weight(1f),
                ) { Text("Pick image") }
            }

            val shown = if (showResult) result else original
            if (shown != null) {
                Image(
                    bitmap = shown.asImageBitmap(),
                    contentDescription = if (showResult) "Translated image" else "Original image",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 160.dp, max = 420.dp),
                )
                if (result != null) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(if (showResult) "Translated" else "Original")
                        Spacer(Modifier.width(8.dp))
                        Switch(checked = showResult, onCheckedChange = { showResult = it })
                    }
                }
            } else {
                Text("Take or pick a photo to translate the text in it.")
            }

            Button(
                onClick = { translate() },
                enabled = original != null && !busy,
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Translate") }

            if (busy) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            status?.let { Text(it) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LanguageDropdown(selected: String, onSelect: (String) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
        OutlinedTextField(
            value = selected,
            onValueChange = {},
            readOnly = true,
            label = { Text("Translate into") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .fillMaxWidth()
                .menuAnchor(),
        )
        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            Settings.LANGUAGES.forEach { lang ->
                DropdownMenuItem(
                    text = { Text(lang) },
                    onClick = {
                        onSelect(lang)
                        expanded = false
                    },
                )
            }
        }
    }
}

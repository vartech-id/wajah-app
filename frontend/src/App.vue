<script setup>
import { ref } from 'vue'

const selectedFile = ref(null)
const preset = ref('cerah')
const originalPreviewUrl = ref(null)
const processedImageUrl = ref(null)
const isLoading = ref(false)
const errorMessage = ref('')

// Sesuaikan dengan URL backend kamu
const API_BASE = 'http://localhost:8000'

const onFileChange = (event) => {
  const file = event.target.files[0]
  if (!file) {
    selectedFile.value = null
    originalPreviewUrl.value = null
    processedImageUrl.value = null
    return
  }

  selectedFile.value = file

  if (originalPreviewUrl.value) {
    URL.revokeObjectURL(originalPreviewUrl.value)
  }
  originalPreviewUrl.value = URL.createObjectURL(file)

  if (processedImageUrl.value) {
    URL.revokeObjectURL(processedImageUrl.value)
    processedImageUrl.value = null
  }

  errorMessage.value = ''
}

const processImage = async () => {
  if (!selectedFile.value) {
    errorMessage.value = 'Silakan upload foto dulu.'
    return
  }

  errorMessage.value = ''
  isLoading.value = true
  processedImageUrl.value = null

  try {
    const formData = new FormData()
    formData.append('image', selectedFile.value)
    formData.append('preset', preset.value)

    const response = await fetch(`${API_BASE}/api/beauty`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || 'Gagal memproses gambar.')
    }

    const blob = await response.blob()

    if (processedImageUrl.value) {
      URL.revokeObjectURL(processedImageUrl.value)
    }
    processedImageUrl.value = URL.createObjectURL(blob)
  } catch (err) {
    console.error(err)
    errorMessage.value = 'Terjadi kesalahan saat memproses gambar.'
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <div>
    <h1>Photobooth Kecantikan</h1>

    <div>
      <label for="fileInput">Upload Foto Wajah:</label>
      <input
        id="fileInput"
        type="file"
        accept="image/*"
        @change="onFileChange"
      />
    </div>

    <div>
      <p>Pilih Preset:</p>
      <label>
        <input
          type="radio"
          value="cerah"
          v-model="preset"
        />
        Cerahkan Kulit
      </label>
      <label>
        <input
          type="radio"
          value="kerutan"
          v-model="preset"
        />
        Kurangi Kerutan / Garis Halus
      </label>
      <label>
        <input
          type="radio"
          value="lembab"
          v-model="preset"
        />
        Lembabkan Kulit
      </label>
    </div>

    <div>
      <button @click="processImage">
        Proses
      </button>
    </div>

    <div v-if="isLoading">
      <p>Memproses gambar, harap tunggu...</p>
    </div>

    <div v-if="errorMessage">
      <p>{{ errorMessage }}</p>
    </div>

    <div v-if="originalPreviewUrl || processedImageUrl">
      <h2>Before / After</h2>
      <div>
        <div>
          <h3>Sebelum</h3>
          <img
            v-if="originalPreviewUrl"
            :src="originalPreviewUrl"
            alt="Before"
          />
        </div>
        <div>
          <h3>Sesudah</h3>
          <img
            v-if="processedImageUrl"
            :src="processedImageUrl"
            alt="After"
          />
        </div>
      </div>
    </div>
  </div>
</template>

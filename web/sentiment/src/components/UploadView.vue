<template>
  <div class="absolute">
    <prime-button
      class="relative top-0 left-0"
      label="< Back"
      @click="router.push('/select')"
    />
  </div>
  <div class="flex justify-content-center">
    <FileUpload
      :customUpload="true"
      @uploader="upload"
      :multiple="false"
    />
  </div>
  <div v-if="transcription != ''" class="flex flex-column align-items-center">
    <h2 v-if="transcription != '***Transcription failed***'">Transcription:</h2>
    <p
      :style="
        sentiment == 'Positive'
          ? 'color: green'
          : sentiment == 'Negative'
          ? 'color: red'
          : 'color: black'
      "
    >
      {{ transcription }}
    </p>
  </div>
  <div
    v-if="sentiment != 'null'"
    class="flex flex-column align-items-center mt-4"
  >
    <h2>Your Sentiment:</h2>
    <img
      v-if="sentiment == 'Positive'"
      src="../assets/happy.png"
      alt="Positive"
      width="120"
      height="120"
    />
    <img
      v-if="sentiment == 'Negative'"
      src="../assets/angry.png"
      alt="Negative"
      width="120"
      height="120"
    />
    <img
      v-if="sentiment == 'Neutral'"
      src="../assets/neutral.png"
      alt="Neutral"
      width="120"
      height="120"
    />
    <p
      :style="
        sentiment == 'Positive'
          ? 'color: green'
          : sentiment == 'Negative'
          ? 'color: red'
          : 'color: black'
      "
    >
      {{ sentimentText }}
    </p>
  </div>
</template>

<script>
import { ref } from "vue";
import { defineComponent } from "@vue/runtime-core";
import { useRouter } from "vue-router";
import axios from "axios";

export default defineComponent({
  name: "UploadView",
  setup() {
    const sentiment = ref("null");
    const sentimentText = ref("");
    const transcription = ref("");
    const router = useRouter();

    const upload = (event) => {
      let data = new FormData();
      data.append("test", event.files[0], "test.wav");
      axios
        .post("http://localhost:5000/sentiment_upload", data, {
          header: {
            "Content-Type": "multipart/form-data",
          },
        })
        .then((result) => {
          if (result.data === null) sentiment.value = "null";
          else sentiment.value = result.data;
          sentimentText.value = result.data;
          transcription.value = result.statusText;
          console.log("Received")
        })
        .catch((error) => {
          console.log(error);
          sentiment.value = "null";
        });
    };

    return {
      upload,
      sentiment,
      sentimentText,
      transcription,
      router,
    };
  },
});
</script>

<style>
</style>
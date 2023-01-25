<template>
  <div class="absolute">
    <prime-button
      class="relative top-0 left-0"
      label="< Back"
      @click="router.push('/select')"
    />
  </div>
  <div class="flex justify-content-center">
    <FileUpload :customUpload="true" @uploader="upload" :multiple="false" />
  </div>
  <div v-if="transcription != ''" class="flex flex-column align-items-center">
    <h2 v-if="transcription != '***Transcription failed***'">Transcription:</h2>
    <p
      v-if="sentimentVader != 'Neutral'"
      :style="
        sentimentVader == 'Positive'
          ? 'color: green'
          : sentimentVader == 'Negative'
          ? 'color: red'
          : 'color: black'
      "
    >
      {{ transcription }}
    </p>
    <p
      v-else
      :style="
        sentimentMLP == 'Positive'
          ? 'color: green'
          : sentimentMLP == 'Negative'
          ? 'color: red'
          : 'color: black'
      "
    >
      {{ transcription }}
    </p>
  </div>
   <div class="flex justify-content-center align-items-center">
    <div class="flex justify-content-center mr-4 align-items-center mr-4">
      <div
        v-if="sentimentVader != 'null'"
        class="flex flex-column align-items-center mr-7"
      >
        <h2>Your Sentiment:</h2>
        <div v-if="sentimentVader == 'Neutral'">
          <img
            v-if="sentimentMLP == 'Positive'"
            src="../assets/happy.png"
            alt="Positive"
            width="120"
            height="120"
          />
          <img
            v-if="sentimentMLP == 'Negative'"
            src="../assets/angry.png"
            alt="Negative"
            width="120"
            height="120"
          />
          <img
            v-if="sentimentMLP == 'Neutral'"
            src="../assets/neutral.png"
            alt="Neutral"
            width="120"
            height="120"
          />
        </div>
        <div v-else>
          <img
            v-if="sentimentVader == 'Positive'"
            src="../assets/happy.png"
            alt="Positive"
            width="120"
            height="120"
          />
          <img
            v-if="sentimentVader == 'Negative'"
            src="../assets/angry.png"
            alt="Negative"
            width="120"
            height="120"
          />
          <img
            v-if="sentimentVader == 'Neutral'"
            src="../assets/neutral.png"
            alt="Neutral"
            width="120"
            height="120"
          />
        </div>
        <p
          :style="
            sentimentVader == 'Positive'
              ? 'color: green'
              : sentimentVader == 'Negative'
              ? 'color: red'
              : 'color: black'
          "
        >
          {{ sentimentText }}
        </p>
      </div>
      <div class="flex flex-column align-items-center" v-if="transcription != ''">
        <div class="flex justify-content-center align-items-center">
          <p class="mr-2">Text model:</p>
          <img
            v-if="sentimentVader == 'Positive'"
            src="../assets/happy.png"
            alt="Positive"
            width="40"
            height="40"
          />
          <img
            v-if="sentimentVader == 'Negative'"
            src="../assets/angry.png"
            alt="Negative"
            width="40"
            height="40"
          />
          <img
            v-if="sentimentVader == 'Neutral'"
            src="../assets/neutral.png"
            alt="Neutral"
            width="40"
            height="40"
          />
        </div>
        <div class="flex justify-content-center">
          <p class="mr-2">Audio model:</p>
          <img
            v-if="sentimentMLP == 'Positive'"
            src="../assets/happy.png"
            alt="Positive"
            width="40"
            height="40"
          />
          <img
            v-if="sentimentMLP == 'Negative'"
            src="../assets/angry.png"
            alt="Negative"
            width="40"
            height="40"
          />
          <img
            v-if="sentimentMLP == 'Neutral'"
            src="../assets/neutral.png"
            alt="Neutral"
            width="40"
            height="40"
          />
        </div>
      </div>
    </div>
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
    const sentimentVader = ref("null");
    const sentimentMLP = ref("null");
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
          if (result.data === null) {
            sentimentVader.value = "null";
            sentimentMLP.value = "null";
          } else {
            sentimentVader.value = result.data[0];
            sentimentMLP.value = result.data[1];
            sentimentText.value =
              sentimentVader.value == "Neutral"
                ? result.data[1]
                : result.data[0];
            transcription.value = result.statusText;
          }
          console.log("Received");
        })
        .catch((error) => {
          console.log(error);
          sentimentVader.value = "null";
          sentimentMLP.value = "null";
        });
    };

    return {
      upload,
      sentimentVader,
      sentimentMLP,
      sentimentText,
      transcription,
      router,
    };
  },
});
</script>

<style>
</style>
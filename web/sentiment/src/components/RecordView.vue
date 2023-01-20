<template>
  <div class="absolute">
    <prime-button
      class="relative top-0 left-0"
      label="< Back"
      @click="router.push('/select')"
    />
  </div>
  <div>
    <prime-button v-if="!recording" :label="label" @click="startRecording" />
    <prime-button v-else label="stop" @click="stopRecording" />
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
import { defineComponent } from "@vue/runtime-core";
import { ref } from "vue";
import axios from "axios";
import getBlobDuration from "get-blob-duration";
import fixWebmDuration from "fix-webm-duration";
import { useRouter } from "vue-router";

export default defineComponent({
  name: "RecordView",
  setup() {
    const recording = ref(false);
    const label = ref("Start Sentiment Recording");
    const sentiment = ref("null");
    const sentimentText = ref("");
    const transcription = ref("");
    const router = useRouter()

    var audioRecorder = {
      audioBlobs: [],
      mediaRecorder: null,
      streamBeingCaptured: null,

      start: function () {
        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
          return Promise.reject(
            new Error(
              "mediaDevices API or getUserMedia method is not supported in this browser."
            )
          );
        } else {
          return navigator.mediaDevices
            .getUserMedia({ audio: true })
            .then((stream) => {
              audioRecorder.streamBeingCaptured = stream;
              const options = {
                audioBitsPerSecond: 128000,
                mimeType: "audio/webm;codec=opus",
              };
              audioRecorder.mediaRecorder = new MediaRecorder(stream, options);

              audioRecorder.audioBlobs = [];

              audioRecorder.mediaRecorder.addEventListener(
                "dataavailable",
                (event) => {
                  audioRecorder.audioBlobs.push(event.data);
                }
              );

              audioRecorder.mediaRecorder.start();
            });
        }
      },
      stop: function () {
        return new Promise((resolve) => {
          let mimeType = audioRecorder.mediaRecorder.mimeType;

          audioRecorder.mediaRecorder.addEventListener("stop", async () => {
            let audioBlob = new Blob(audioRecorder.audioBlobs, {
              type: mimeType,
            });
            const duration = await getBlobDuration(audioBlob);
            fixWebmDuration(audioBlob, duration, { logger: false }).then(
              function (fixedBlob) {
                resolve(fixedBlob);
              }
            );
          });
          audioRecorder.cancel();
        });
      },
      cancel: function () {
        audioRecorder.mediaRecorder.stop();
        audioRecorder.stopStream();
        audioRecorder.resetRecordingProperties();
      },
      stopStream: function () {
        audioRecorder.streamBeingCaptured
          .getTracks()
          .forEach((track) => track.stop());
      },
      resetRecordingProperties: function () {
        audioRecorder.mediaRecorder = null;
        audioRecorder.streamBeingCaptured = null;
      },
    };

    const startRecording = () => {
      sentiment.value = "null";
      sentimentText.value = "";
      transcription.value = "";
      audioRecorder
        .start()
        .then(() => {
          console.log("Recording Audio...");
          recording.value = true;
        })
        .catch((error) => {
          if (
            error.message.includes(
              "mediaDevices API or getUserMedia method is not supported in this browser."
            )
          ) {
            console.log(
              "To record audio, use browsers like Chrome and Firefox."
            );
            //Error handling structure
            switch (error.name) {
              case "AbortError": //error from navigator.mediaDevices.getUserMedia
                console.log("An AbortError has occured.");
                break;
              case "NotAllowedError": //error from navigator.mediaDevices.getUserMedia
                console.log(
                  "A NotAllowedError has occured. User might have denied permission."
                );
                break;
              case "NotFoundError": //error from navigator.mediaDevices.getUserMedia
                console.log("A NotFoundError has occured.");
                break;
              case "NotReadableError": //error from navigator.mediaDevices.getUserMedia
                console.log("A NotReadableError has occured.");
                break;
              case "SecurityError": //error from navigator.mediaDevices.getUserMedia or from the MediaRecorder.start
                console.log("A SecurityError has occured.");
                break;
              case "TypeError": //error from navigator.mediaDevices.getUserMedia
                console.log("A TypeError has occured.");
                break;
              case "InvalidStateError": //error from the MediaRecorder.start
                console.log("An InvalidStateError has occured.");
                break;
              case "UnknownError": //error from the MediaRecorder.start
                console.log("An UnknownError has occured.");
                break;
              default:
                console.log(
                  "An error occured with the error name " + error.name
                );
            }
          }
        });
    };

    const stopRecording = () => {
      console.log("Stopping Audio Recording...");
      recording.value = false;
      label.value = "Record sentiment again";
      audioRecorder
        .stop()
        .then((audioAsblob) => {
          console.log("stopped with audio Blob:", audioAsblob);
          let data = new FormData();
          data.append("test", audioAsblob, "test.webm");
          axios
            .post("http://localhost:5000/sentiment", data, {
              header: {
                "Content-Type": "multipart/form-data",
              },
            })
            .then((result) => {
              if (result.data === null) sentiment.value = "null";
              else sentiment.value = result.data;
              sentimentText.value = result.data;
              transcription.value = result.statusText;
            })
            .catch((error) => {
              console.log(error);
              sentiment.value = "null";
            });
        })
        .catch((error) => {
          switch (error.name) {
            case "InvalidStateError":
              console.log("An InvalidStateError has occured.");
              break;
            default:
              console.log("An error occured with the error name " + error.name);
          }
        });
    };

    const cancelRecording = () => {
      console.log("Canceling audio...");
      audioRecorder.cancel();
    };

    return {
      recording,
      label,
      sentiment,
      sentimentText,
      transcription,
      router,
      startRecording,
      stopRecording,
      cancelRecording,
    };
  },
});
</script>

<style>
</style>
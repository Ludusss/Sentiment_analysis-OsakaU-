<template>
  <div class="absolute">
    <prime-button class="relative top-0 left-0" label="< Back" @click="router.push('/select')" />
  </div>
  <div>
    <prime-button v-if="!recording" :label="label" @click="startRecording" />
    <prime-button v-else label="stop" @click="stopRecording" />
  </div>
  <div v-if="transcription != ''" class="flex flex-column align-items-center">
    <h2 v-if="transcription != '***Transcription failed***'">Transcription:</h2>
    <p v-if="sentimentVader != 'Neutral'" :style="
      sentimentVader == 'Positive'
        ? 'color: green'
        : sentimentVader == 'Negative'
          ? 'color: red'
          : 'color: black'
    ">
      {{ transcription }}
    </p>
    <p v-else :style="
      sentimentMLP == 'Positive'
        ? 'color: green'
        : sentimentMLP == 'Negative'
          ? 'color: red'
          : 'color: black'
    ">
      {{ transcription }}
    </p>
  </div>
  <div v-if="transcription != '***Transcription failed***'" class="flex justify-content-center align-items-center">
    <div class="flex justify-content-center mr-4 align-items-center mr-4">
      <div v-if="sentimentVader != 'null'" class="flex flex-column align-items-center mr-7">
        <h2>Your Sentiment:</h2>
        <div v-if="sentimentVader == 'Neutral' && sentimentMLP != 'Failed'">
          <img v-if="sentimentMLP == 'Positive'" src="../assets/happy.png" alt="Positive" width="120" height="120" />
          <img v-if="sentimentMLP == 'Negative'" src="../assets/angry.png" alt="Negative" width="120" height="120" />
          <img v-if="sentimentMLP == 'Neutral'" src="../assets/neutral.png" alt="Neutral" width="120" height="120" />
        </div>
        <div v-else>
          <img v-if="sentimentVader == 'Positive'" src="../assets/happy.png" alt="Positive" width="120" height="120" />
          <img v-if="sentimentVader == 'Negative'" src="../assets/angry.png" alt="Negative" width="120" height="120" />
          <img v-if="sentimentVader == 'Neutral'" src="../assets/neutral.png" alt="Neutral" width="120" height="120" />
        </div>
        <p>
          {{ sentimentText }}
        </p>
      </div>
      <div class="flex flex-column align-items-center" v-if="transcription != ''">
        <div class="flex justify-content-center align-items-center">
          <p class="mr-2">Text model:</p>
          <img v-if="sentimentVader == 'Positive'" src="../assets/happy.png" alt="Positive" width="40" height="40" />
          <img v-if="sentimentVader == 'Negative'" src="../assets/angry.png" alt="Negative" width="40" height="40" />
          <img v-if="sentimentVader == 'Neutral'" src="../assets/neutral.png" alt="Neutral" width="40" height="40" />
        </div>
        <div class="flex justify-content-center">
          <p class="mr-2">Audio model:</p>
          <p v-if="sentimentMLP == 'not used'">Not used</p>
          <p v-if="sentimentMLP == 'Failed'">Failed</p>
          <img v-if="sentimentMLP == 'Positive'" src="../assets/happy.png" alt="Positive" width="40" height="40" />
          <img v-if="sentimentMLP == 'Negative'" src="../assets/angry.png" alt="Negative" width="40" height="40" />
          <img v-if="sentimentMLP == 'Neutral'" src="../assets/neutral.png" alt="Neutral" width="40" height="40" />
        </div>
      </div>
    </div>
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
    const sentimentVader = ref("null");
    const sentimentMLP = ref("null");
    const sentimentText = ref("");
    const transcription = ref("");
    const router = useRouter();

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
      sentimentVader.value = "null";
      sentimentMLP.value = "null";
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
              if (result.data === null) {
                sentimentVader.value = "null";
                sentimentMLP.value = "null";
              } else {
                sentimentVader.value = result.data[0];
                sentimentMLP.value = result.data[1];
                sentimentText.value =
                  sentimentVader.value == "Neutral"
                    ? result.data[1] != "Failed" ? result.data[1] : result.data[0]
                    : result.data[0];
              }
              transcription.value = result.statusText;
            })
            .catch((error) => {
              console.log(error);
              sentimentVader.value = "null";
              sentimentMLP.value = "null";
              transcription.value = "***Transcription failed***";
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
      sentimentVader,
      sentimentMLP,
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
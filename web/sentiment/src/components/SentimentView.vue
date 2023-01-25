<template>
  <div class="flex flex-column align-items-center">
    <h1>Welcome to the Sentiment Analyzer</h1>
    <div class="flex flex-column">
      <prime-button label="Start" @click="start" />
    </div>
  </div>
</template>

<script>

import { useRouter } from 'vue-router';

export default {
  name: "SentimentView",
  setup() {
    const router = useRouter()

    const start = () => {
      router.push("/select")
    }
    return {
      start,
    };
  },
};
</script>

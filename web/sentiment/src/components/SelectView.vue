<template>
  <div class="flex justify-content-center mb-5">
    <prime-button class="mr-4" label="Record mode" @click="router.push('/record')" />
    <prime-button label="Upload mode" @click="router.push('/upload')" />
  </div>
  <prime-button label="Back" @click="router.push('/')" />
</template>

<script>
import { defineComponent } from "@vue/runtime-core";
import { useRouter } from "vue-router";

export default defineComponent({
    name:"SelectView",
    setup(){
      const router = useRouter();

      return {
        router
      }

    }


})
</script>

<style>

</style>
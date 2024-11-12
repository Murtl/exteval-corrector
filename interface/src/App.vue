<script setup lang="ts">
import { ref } from 'vue'

const name = ref('')
</script>

<template>
  <header>
    Welcome!
  </header>

  <main>
    <input type="text" v-model="name" />
  </main>
</template>

<style scoped lang="scss">
header {
  background-color: #333;
  color: white;
  padding: 1rem;
  text-align: center;
}

main {
  padding: 1rem;
  text-align: center;
}
</style>

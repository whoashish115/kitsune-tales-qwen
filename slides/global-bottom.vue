<template>
  <div v-if="$nav.currentPage > 1" class="absolute bottom-3 left-14 right-14 flex justify-between text-xs" style="color: #71717a">
    <span>Kitsune Tales</span>
    <span>{{ $nav.currentPage }} / {{ $nav.total }}</span>
  </div>
</template>

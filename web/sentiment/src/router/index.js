import { createRouter, createWebHistory } from 'vue-router'
import Select from "../components/SelectView.vue";
import Upload from "../components/UploadView.vue";
import Record from "../components/RecordView.vue";
import SentimentView from "../components/SentimentView.vue";

const routes = [
  { path: '/', name:"home", component: SentimentView },
  { path: '/select', name:"select", component: Select },
  { path: '/record', name:"record", component: Record },
  { path: '/upload', name:"upload", component: Upload },
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router

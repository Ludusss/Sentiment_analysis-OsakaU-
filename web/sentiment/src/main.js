import { createApp } from 'vue'
import App from './App.vue'
import PrimeVue from "primevue/config";
import Button from 'primevue/button';
import FileUpload from 'primevue/fileupload';
import "primeflex/primeflex.css";
import "primevue/resources/themes/saga-blue/theme.css"       //theme
import "primevue/resources/primevue.min.css"                 //core css
import "primeicons/primeicons.css"                           //icons
import router from './router'

const app = createApp(App).use(router);
app.use(PrimeVue, { ripple: true, inputStyle: "filled" });
app.component("PrimeButton", Button);
app.component("FileUpload", FileUpload);
app.use(router)
app.mount('#app');

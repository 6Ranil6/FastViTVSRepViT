plugins { id("com.android.application") }
android {
    namespace = "ru.bmstu.flowers"
    compileSdk = 35
    defaultConfig {
        applicationId = "ru.bmstu.flowers"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources { noCompress += listOf("pte", "rgb") }
    buildTypes { release { isMinifyEnabled = false } }
}
dependencies { implementation("org.pytorch:executorch-android:1.0.0") }

val verifyModelAssets by tasks.registering {
    doLast {
        listOf("manifest.json", "fastvit.pte", "repvit.pte").forEach {
            check(file("src/main/assets/bundle/$it").isFile) {
                "Missing model assets. Run python -m coursework.bundle from the project root."
            }
        }
    }
}
tasks.named("preBuild") { dependsOn(verifyModelAssets) }

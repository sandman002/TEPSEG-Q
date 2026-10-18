plugins {
    id("com.gradleup.shadow") version "8.3.5"
    id("qupath-conventions")
}

qupathExtension {
    name = "qupath-extension-tepseg"
    group = "io.github.tepseg"
    version = "1.0.0"
    description = "Segment tumour epithelium regions in H&E whole-slide images (TEPSEG, UNI2-h)"
    automaticModule = "io.github.tepseg.extension"
}

dependencies {
    // Everything we use ships with QuPath, so the jar has no bundled dependencies.
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)

    testImplementation(libs.bundles.qupath)
    testImplementation(libs.junit)
}

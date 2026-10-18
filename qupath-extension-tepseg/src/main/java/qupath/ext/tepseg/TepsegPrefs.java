package qupath.ext.tepseg;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.StringProperty;
import qupath.lib.gui.prefs.PathPrefs;

import java.nio.file.Path;

/** Persistent settings of the TEPSEG extension. */
public final class TepsegPrefs {

    private TepsegPrefs() {}

    /** Folder holding uv, the managed Python and the TEPSEG environment. */
    public static final StringProperty envDir = PathPrefs.createPersistentPreference(
            "tepseg.envDir", Path.of(System.getProperty("user.home"), ".tepseg").toString());

    /** Optional existing Python executable (with tepseg installed); overrides the managed environment. */
    public static final StringProperty pythonOverride = PathPrefs.createPersistentPreference("tepseg.python", "");

    /** What to install: a git URL (default), a wheel path or a requirement. */
    public static final StringProperty packageSpec = PathPrefs.createPersistentPreference("tepseg.packageSpec",
            "tepseg @ git+https://github.com/sandman002/TEPSEG-Q.git");

    /** Spacing of the scored 128 um fields: "128" (standard) or "256" (fast, ~4x fewer fields). */
    public static final StringProperty strideUm = PathPrefs.createPersistentPreference("tepseg.strideUm", "128");

    /** Tumour probability cut-off. */
    public static final DoubleProperty threshold = PathPrefs.createPersistentPreference("tepseg.threshold", 0.5);

    /** Smaller tumour regions are dropped (mm²). */
    public static final DoubleProperty minAreaMm2 = PathPrefs.createPersistentPreference("tepseg.minAreaMm2", 0.5);

    /** Output folder; empty = "tepseg" inside the project folder (or next to the image without a project). */
    public static final StringProperty outputDir = PathPrefs.createPersistentPreference("tepseg.outputDir", "");

    /** Skip images that already have results in the output folder. */
    public static final BooleanProperty skipExisting = PathPrefs.createPersistentPreference("tepseg.skipExisting", false);

    /** Optional local UNI2-h pytorch_model.bin; empty = Hugging Face (or an imported copy). */
    public static final StringProperty uni2Path = PathPrefs.createPersistentPreference("tepseg.uni2", "");
}

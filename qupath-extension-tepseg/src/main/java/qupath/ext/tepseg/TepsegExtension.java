package qupath.ext.tepseg;

import javafx.scene.control.MenuItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.tepseg.ui.SetupWindow;
import qupath.ext.tepseg.ui.TepsegWindow;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;

/** Adds Extensions › TEPSEG to QuPath. */
public class TepsegExtension implements QuPathExtension, GitHubProject {

    private static final Logger logger = LoggerFactory.getLogger(TepsegExtension.class);

    private boolean installed;
    private TepsegWindow window;

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (installed)
            return;
        installed = true;
        var menu = qupath.getMenu("Extensions>TEPSEG", true);
        var open = new MenuItem("Segment tumour…");
        open.setOnAction(e -> {
            if (window == null)
                window = new TepsegWindow(qupath);
            window.show();
        });
        var setup = new MenuItem("Set up backend…");
        setup.setOnAction(e -> new SetupWindow(qupath, () -> {}).show());
        var docs = new MenuItem("Documentation");
        docs.setOnAction(e -> QuPathGUI.openInBrowser("https://github.com/sandman002/TEPSEG-Q"));
        menu.getItems().addAll(open, setup, docs);
        logger.info("TEPSEG extension installed");
    }

    @Override
    public String getName() {
        return "TEPSEG";
    }

    @Override
    public String getDescription() {
        return "Segments tumour epithelium regions in H&E whole-slide images (UNI2-h foundation model, GPU), "
                + "for one image or a whole project.";
    }

    @Override
    public Version getQuPathVersion() {
        return Version.parse("0.7.0");
    }

    @Override
    public GitHubRepo getRepository() {
        return GitHubRepo.create(getName(), "sandman002", "TEPSEG-Q");
    }
}

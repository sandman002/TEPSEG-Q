package qupath.ext.tepseg.ui;

import com.google.gson.JsonObject;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.PasswordField;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.TitledPane;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import qupath.ext.tepseg.TepsegPrefs;
import qupath.ext.tepseg.env.EnvironmentManager;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;

import java.io.File;
import java.nio.file.Path;

/**
 * One-click installation of the Python backend (uv + Python + TEPSEG + GPU
 * PyTorch), and access to the gated UNI2-h model: a Hugging Face sign-in, or
 * a local copy of its weights.
 */
public class SetupWindow {

    static final String UNI2_PAGE = "https://huggingface.co/MahmoodLab/UNI2-h";
    static final String TOKEN_PAGE = "https://huggingface.co/settings/tokens";
    private static final String[] STEPS = {"Download uv", "Install Python", "Install TEPSEG + PyTorch",
            "Download UNI2-h", "Check GPU"};

    private final Stage stage = new Stage();
    private final Runnable onInstalled;
    private final TextArea log = new TextArea();
    private final ProgressBar bar = new ProgressBar(0);
    private final Label stepLabel = new Label("Not started");
    private final Label[] stepMarks = new Label[STEPS.length];
    private final Button install = new Button("Install / repair");
    private final Button test = new Button("Test backend");
    private final Label uni2Status = new Label("UNI2-h: not checked yet");
    private final PasswordField token = new PasswordField();
    private final Button signIn = new Button("Sign in");

    public SetupWindow(QuPathGUI qupath, Runnable onInstalled) {
        this.onInstalled = onInstalled;
        stage.initOwner(qupath.getStage());
        stage.setTitle("TEPSEG – set up backend");
        stage.setScene(new Scene(build(), 780, 780));
    }

    public void show() {
        stage.show();
        stage.toFront();
        if (new EnvironmentManager().isReady())
            refreshStatus();
    }

    private Region build() {
        var intro = new Label("TEPSEG runs its model in Python. This installs a private, self-contained environment "
                + "(Python 3.11, PyTorch with GPU support and TEPSEG). Nothing else on your computer is changed; "
                + "delete the folder below to uninstall. Needs ~6 GB of disk and an internet connection.");
        intro.setWrapText(true);

        var folder = new TextField();
        folder.textProperty().bindBidirectional(TepsegPrefs.envDir);
        HBox.setHgrow(folder, Priority.ALWAYS);
        var browse = new Button("Browse…");
        browse.setOnAction(e -> {
            File f = new DirectoryChooser().showDialog(stage);
            if (f != null) TepsegPrefs.envDir.set(f.getAbsolutePath());
        });
        var folderRow = new HBox(6, folder, browse);
        HBox.setHgrow(folderRow, Priority.ALWAYS);
        var grid = new HBox(10, new Label("Install folder"), folderRow);
        grid.setAlignment(Pos.CENTER_LEFT);

        var steps = new VBox(4);
        for (int i = 0; i < STEPS.length; i++) {
            stepMarks[i] = new Label("○  " + STEPS[i]);
            steps.getChildren().add(stepMarks[i]);
        }
        bar.setMaxWidth(Double.MAX_VALUE);
        var progress = new VBox(6, stepLabel, bar);
        HBox.setHgrow(progress, Priority.ALWAYS);

        install.setDefaultButton(true);
        install.setOnAction(e -> runInstall());
        test.setOnAction(e -> refreshStatus());
        var close = new Button("Close");
        close.setOnAction(e -> stage.close());
        var spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        var buttons = new HBox(10, install, test, spacer, close);
        buttons.setAlignment(Pos.CENTER_LEFT);

        log.setEditable(false);
        log.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        VBox.setVgrow(log, Priority.ALWAYS);

        var root = new VBox(12, intro, grid, new HBox(24, steps, progress), buttons, uni2Pane(), advanced(), log);
        root.setPadding(new Insets(14));
        return root;
    }

    /** UNI2-h is gated: request access, then sign in with a read token, or point to a local copy. */
    private Region uni2Pane() {
        var text = new Label("TEPSEG uses UNI2-h (MahmoodLab), a gated model on Hugging Face. Once per user: "
                + "1. request access on its page, 2. create a Read token, 3. paste it here and sign in. "
                + "The token is kept by Hugging Face's own tools on this computer, not by QuPath.");
        text.setWrapText(true);
        var page = new Button("1. Request access…");
        page.setOnAction(e -> QuPathGUI.openInBrowser(UNI2_PAGE));
        var tokens = new Button("2. Create token…");
        tokens.setOnAction(e -> QuPathGUI.openInBrowser(TOKEN_PAGE));
        token.setPromptText("3. paste the token (hf_…)");
        HBox.setHgrow(token, Priority.ALWAYS);
        signIn.setOnAction(e -> runSignIn());
        token.setOnAction(e -> runSignIn());

        var local = new TextField();
        local.textProperty().bindBidirectional(TepsegPrefs.uni2Path);
        local.setPromptText("optional: an existing UNI2-h pytorch_model.bin, used instead of downloading");
        HBox.setHgrow(local, Priority.ALWAYS);
        var pick = new Button("Browse…");
        pick.setOnAction(e -> {
            var fc = new FileChooser();
            fc.setTitle("UNI2-h pytorch_model.bin");
            fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("PyTorch weights", "*.bin", "*.pth", "*.pt"));
            File f = fc.showOpenDialog(stage);
            if (f != null) {
                TepsegPrefs.uni2Path.set(f.getAbsolutePath());
                uni2Status.setText("UNI2-h: local file " + f.getName());
            }
        });
        uni2Status.setStyle("-fx-font-weight: bold;");
        var localRow = new HBox(8, new Label("Local copy"), local, pick);
        localRow.setAlignment(Pos.CENTER_LEFT);
        var box = new VBox(8, text, new HBox(8, page, tokens), new HBox(8, token, signIn), localRow, uni2Status);
        box.setPadding(new Insets(8));
        var pane = new TitledPane("UNI2-h model access", box);
        pane.setCollapsible(false);
        return pane;
    }

    private Region advanced() {
        var python = new TextField();
        python.textProperty().bindBidirectional(TepsegPrefs.pythonOverride);
        python.setPromptText("empty = use the managed environment");
        HBox.setHgrow(python, Priority.ALWAYS);
        var browse = new Button("Browse…");
        browse.setOnAction(e -> {
            var fc = new FileChooser();
            fc.setTitle("Python executable with tepseg installed");
            File f = fc.showOpenDialog(stage);
            if (f != null) TepsegPrefs.pythonOverride.set(f.getAbsolutePath());
        });
        var spec = new TextField();
        spec.textProperty().bindBidirectional(TepsegPrefs.packageSpec);
        spec.setTooltip(new Tooltip("What to install: a git URL (default: the TEPSEG-Q GitHub repository), a wheel file, "
                + "or a git tag ('tepseg @ git+https://github.com/sandman002/TEPSEG-Q.git@v1.0.0')"));
        var grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(8);
        grid.setPadding(new Insets(8));
        grid.addRow(0, new Label("Existing Python"), new HBox(6, python, browse));
        grid.addRow(1, new Label("Package"), spec);
        var pane = new TitledPane("Advanced", grid);
        pane.setExpanded(false);
        return pane;
    }

    private void append(String line) {
        Platform.runLater(() -> log.appendText(line + "\n"));
    }

    private void runInstall() {
        install.setDisable(true);
        test.setDisable(true);
        log.clear();
        bar.setProgress(-1);
        var env = new EnvironmentManager(Path.of(TepsegPrefs.envDir.get()));
        Thread t = new Thread(() -> {
            try {
                env.install(this::append, (i, name) -> Platform.runLater(() -> markStep(i, name)));
                Platform.runLater(() -> {
                    markStep(STEPS.length, "Installed");
                    bar.setProgress(1);
                    stepLabel.setText("✓ Installed");
                    onInstalled.run();
                });
                refreshStatus();
            } catch (Exception ex) {
                Platform.runLater(() -> {
                    bar.setProgress(0);
                    stepLabel.setText("✗ " + ex.getMessage());
                    log.appendText("\nFAILED: " + ex.getMessage() + "\n");
                    Dialogs.showErrorMessage("TEPSEG setup", ex.getMessage() + "\n\nSee the log for details.");
                });
            } finally {
                Platform.runLater(() -> {
                    install.setDisable(false);
                    test.setDisable(false);
                });
            }
        }, "tepseg-install");
        t.setDaemon(true);
        t.start();
    }

    private void markStep(int index, String name) {
        for (int i = 0; i < STEPS.length; i++)
            stepMarks[i].setText((i < index ? "✓  " : i == index ? "▶  " : "○  ") + STEPS[i]);
        if (index < STEPS.length) {
            stepLabel.setText(name + "…");
            bar.setProgress(-1);
        }
    }

    private void runSignIn() {
        String t = token.getText().strip();
        if (t.isEmpty())
            return;
        var env = new EnvironmentManager();
        if (!env.isReady()) {
            Dialogs.showWarningNotification("TEPSEG", "Install the backend first");
            return;
        }
        signIn.setDisable(true);
        uni2Status.setText("UNI2-h: signing in…");
        Thread th = new Thread(() -> {
            try {
                JsonObject r = EnvironmentManager.tepseg(env.python(), this::append, t, "login");
                boolean ok = r.has("ok") && r.get("ok").getAsBoolean();
                String user = r.has("user") && !r.get("user").isJsonNull() ? r.get("user").getAsString() : null;
                String msg = r.has("message") ? r.get("message").getAsString() : "";
                Platform.runLater(() -> {
                    token.clear();
                    uni2Status.setText("UNI2-h: " + (user != null ? "signed in as " + user + " · " : "") + msg);
                });
                if (ok) {
                    Platform.runLater(() -> uni2Status.setText("UNI2-h: downloading (2.7 GB, once)…"));
                    JsonObject f = EnvironmentManager.tepseg(env.python(), this::append, null, "fetch-weights");
                    append(f.toString());
                    refreshStatus();
                }
            } catch (Exception ex) {
                Platform.runLater(() -> uni2Status.setText("UNI2-h: sign-in failed (" + ex.getMessage() + ")"));
            } finally {
                Platform.runLater(() -> signIn.setDisable(false));
            }
        }, "tepseg-login");
        th.setDaemon(true);
        th.start();
    }

    private void refreshStatus() {
        var env = new EnvironmentManager();
        Thread t = new Thread(() -> {
            try {
                var info = EnvironmentManager.info(env.python(), this::append);
                String uni2 = describeUni2(info);
                Platform.runLater(() -> {
                    stepLabel.setText("✓ " + EnvironmentManager.describe(info));
                    uni2Status.setText(uni2);
                });
            } catch (Exception ex) {
                append("✗ " + ex.getMessage());
            }
        }, "tepseg-test");
        t.setDaemon(true);
        t.start();
    }

    /** One line about UNI2-h for the user, from {@code tepseg info}. */
    public static String describeUni2(JsonObject info) {
        if (!TepsegPrefs.uni2Path.get().isBlank())
            return "UNI2-h: local file " + Path.of(TepsegPrefs.uni2Path.get()).getFileName();
        if (info.has("uni2") && !info.get("uni2").isJsonNull())
            return "UNI2-h: ready ✓";
        String user = info.has("hf_user") && !info.get("hf_user").isJsonNull() ? info.get("hf_user").getAsString() : null;
        if (user == null)
            return "UNI2-h: not signed in — follow steps 1–3 above";
        String msg = info.has("uni2_message") ? info.get("uni2_message").getAsString() : "not downloaded yet";
        return "UNI2-h: signed in as " + user + " · " + msg;
    }
}

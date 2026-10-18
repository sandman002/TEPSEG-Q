package qupath.ext.tepseg.ui;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.RadioButton;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableRow;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.TitledPane;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.tepseg.TepsegPrefs;
import qupath.ext.tepseg.env.EnvironmentManager;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.io.PathIO;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

import java.awt.Desktop;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Segment tumour regions on the current image or on project images. Runs
 * {@code python -m tepseg run --json}, fills a live results table and imports
 * each image's regions as "Tumor" annotations (named "TEPSEG") as soon as it finishes.
 */
public class TepsegWindow {

    private static final Logger logger = LoggerFactory.getLogger(TepsegWindow.class);
    static final String NAME = "TEPSEG";
    static final String CLASS = "Tumor";

    private record Target(String name, Path slide, ImageData<?> open, ProjectImageEntry<?> entry) {}

    /** One line of the results table. */
    public static class Row {
        final StringProperty image = new SimpleStringProperty();
        final StringProperty status = new SimpleStringProperty("Queued");
        final StringProperty tumour = new SimpleStringProperty("");
        final StringProperty tissue = new SimpleStringProperty("");
        final StringProperty fraction = new SimpleStringProperty("");
        final StringProperty regions = new SimpleStringProperty("");
        final StringProperty time = new SimpleStringProperty("");
        Target target;

        Row(Target t) {
            target = t;
            image.set(t.name());
        }
    }

    private final QuPathGUI qupath;
    private final Stage stage = new Stage();
    private final RadioButton scopeCurrent = new RadioButton("Current image");
    private final RadioButton scopeProject = new RadioButton("Project images");
    private final Label chosenLabel = new Label("all images");
    private final Button run = new Button("Segment tumour");
    private final Button cancel = new Button("Cancel");
    private final ProgressBar bar = new ProgressBar(0);
    private final Label status = new Label("Ready");
    private final Circle dot = new Circle(5, Color.GRAY);
    private final Label backend = new Label("Backend: checking…");
    private final TextArea log = new TextArea();
    private final ObservableList<Row> rows = FXCollections.observableArrayList();
    private List<ProjectImageEntry<?>> chosen = new ArrayList<>();
    private volatile Process process;
    private Path lastOutput;

    public TepsegWindow(QuPathGUI qupath) {
        this.qupath = qupath;
        stage.initOwner(qupath.getStage());
        stage.setTitle("TEPSEG");
        stage.setScene(new Scene(build(), 820, 760));
    }

    public void show() {
        stage.show();
        stage.toFront();
        checkBackend();
    }

    // ------------------------------------------------------------------ UI

    private Region build() {
        var title = new Label("TEPSEG");
        title.setStyle("-fx-font-size: 22px; -fx-font-weight: bold;");
        var subtitle = new Label("Tumour epithelium segmentation in H&E · UNI2-h foundation model");
        subtitle.setStyle("-fx-text-fill: -fx-mid-text-color;");
        var setup = new Button("Set up…");
        setup.setOnAction(e -> new SetupWindow(qupath, this::checkBackend).show());
        var spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        var backendBox = new HBox(6, dot, backend);
        backendBox.setAlignment(Pos.CENTER_RIGHT);
        var header = new HBox(10, new VBox(2, title, subtitle), spacer, new VBox(6, backendBox, setup));
        ((VBox) header.getChildren().get(2)).setAlignment(Pos.TOP_RIGHT);

        var root = new VBox(12, header, section("RUN ON", scope()), section("SETTINGS", settings()),
                actions(), new VBox(4, status, bar), table(), logPane());
        root.setPadding(new Insets(14));
        VBox.setVgrow(root.getChildren().get(5), Priority.ALWAYS);
        return root;
    }

    private static Region section(String name, Region content) {
        var label = new Label(name);
        label.setStyle("-fx-font-size: 11px; -fx-font-weight: bold; -fx-text-fill: -fx-mid-text-color;");
        var box = new VBox(6, label, content);
        box.setPadding(new Insets(10));
        box.setStyle("-fx-background-color: derive(-fx-base, 8%); -fx-background-radius: 6; "
                + "-fx-border-color: derive(-fx-base, -12%); -fx-border-radius: 6;");
        return box;
    }

    private Region scope() {
        var tg = new ToggleGroup();
        scopeCurrent.setToggleGroup(tg);
        scopeProject.setToggleGroup(tg);
        scopeCurrent.setSelected(true);
        var choose = new Button("Choose…");
        choose.disableProperty().bind(scopeProject.selectedProperty().not());
        chosenLabel.disableProperty().bind(scopeProject.selectedProperty().not());
        choose.setOnAction(e -> {
            Project<?> project = qupath.getProject();
            if (project == null) {
                Dialogs.showWarningNotification(NAME, "Open a project first");
                return;
            }
            var picked = ImageChooser.choose(stage, project, chosen);
            if (picked != null) {
                chosen = new ArrayList<>(picked);
                chosenLabel.setText(chosen.isEmpty() || chosen.size() == project.getImageList().size() ? "all images"
                        : chosen.size() + " image(s)");
            }
        });
        var box = new HBox(16, scopeCurrent, scopeProject, choose, chosenLabel);
        box.setAlignment(Pos.CENTER_LEFT);
        return box;
    }

    private Region settings() {
        var mode = new ChoiceBox<>(FXCollections.observableArrayList(
                "Standard: every 128 µm field", "Fast: every other field (~4× faster)"));
        mode.getSelectionModel().select("256".equals(TepsegPrefs.strideUm.get()) ? 1 : 0);
        mode.getSelectionModel().selectedIndexProperty().addListener((o, a, v) ->
                TepsegPrefs.strideUm.set(v.intValue() == 1 ? "256" : "128"));

        var thr = new Slider(0.2, 0.9, TepsegPrefs.threshold.get());
        thr.setMajorTickUnit(0.1);
        thr.setBlockIncrement(0.05);
        thr.setPrefWidth(220);
        var thrLabel = new Label();
        thrLabel.textProperty().bind(thr.valueProperty().asString(Locale.ROOT, "%.2f"));
        thr.valueProperty().addListener((o, a, v) -> TepsegPrefs.threshold.set(Math.round(v.doubleValue() * 100) / 100.0));
        var thrBox = new HBox(8, thr, thrLabel, hint("lower = more tumour"));
        thrBox.setAlignment(Pos.CENTER_LEFT);

        var minArea = new Spinner<Double>(0.0, 50.0, TepsegPrefs.minAreaMm2.get(), 0.1);
        minArea.setEditable(true);
        minArea.setPrefWidth(90);
        minArea.valueProperty().addListener((o, a, v) -> TepsegPrefs.minAreaMm2.set(v));
        var areaBox = new HBox(8, minArea, new Label("mm²"), hint("smaller regions are dropped"));
        areaBox.setAlignment(Pos.CENTER_LEFT);

        var skip = new CheckBox("Skip images already segmented (reuse their results)");
        skip.selectedProperty().bindBidirectional(TepsegPrefs.skipExisting);

        var out = new TextField();
        out.textProperty().bindBidirectional(TepsegPrefs.outputDir);
        out.setPromptText("default: <project folder>/tepseg");
        HBox.setHgrow(out, Priority.ALWAYS);
        var browse = new Button("Browse…");
        browse.setOnAction(e -> {
            File f = new DirectoryChooser().showDialog(stage);
            if (f != null) TepsegPrefs.outputDir.set(f.getAbsolutePath());
        });

        var grid = new GridPane();
        grid.setHgap(12);
        grid.setVgap(8);
        int r = 0;
        grid.addRow(r++, new Label("Mode"), mode);
        grid.addRow(r++, new Label("Threshold"), thrBox);
        grid.addRow(r++, new Label("Min. region"), areaBox);
        grid.addRow(r++, new Label(""), skip);
        var outBox = new HBox(6, out, browse);
        grid.addRow(r, new Label("Output folder"), outBox);
        GridPane.setHgrow(outBox, Priority.ALWAYS);
        return grid;
    }

    private static Label hint(String s) {
        var l = new Label(s);
        l.setStyle("-fx-text-fill: -fx-mid-text-color; -fx-font-size: 11px;");
        return l;
    }

    private Region actions() {
        run.setDefaultButton(true);
        run.setStyle("-fx-font-weight: bold;");
        run.setOnAction(e -> start());
        cancel.setDisable(true);
        cancel.setOnAction(e -> stop());
        var openOut = new Button("Open output folder");
        openOut.setOnAction(e -> openPath(lastOutput != null ? lastOutput : outputDirGuess()));
        var openSummary = new Button("Open summary.csv");
        openSummary.setOnAction(e -> {
            Path base = lastOutput != null ? lastOutput : outputDirGuess();
            if (base != null)
                openPath(base.resolve("summary.csv"));
        });
        var spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        var box = new HBox(10, run, cancel, spacer, openOut, openSummary);
        box.setAlignment(Pos.CENTER_LEFT);
        bar.setMaxWidth(Double.MAX_VALUE);
        return box;
    }

    private Region table() {
        var t = new TableView<>(rows);
        t.setPlaceholder(new Label("Results appear here, one row per image"));
        t.getColumns().add(col("Image", r -> r.image, 220));
        t.getColumns().add(col("Status", r -> r.status, 130));
        t.getColumns().add(col("Tumour mm²", r -> r.tumour, 90));
        t.getColumns().add(col("Tissue mm²", r -> r.tissue, 90));
        t.getColumns().add(col("Tumour %", r -> r.fraction, 75));
        t.getColumns().add(col("Regions", r -> r.regions, 65));
        t.getColumns().add(col("Time", r -> r.time, 60));
        t.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_FLEX_LAST_COLUMN);
        t.setRowFactory(tv -> {
            var row = new TableRow<Row>();
            row.setOnMouseClicked(e -> {
                if (e.getClickCount() == 2 && !row.isEmpty() && row.getItem().target.entry() != null)
                    openEntry(row.getItem().target.entry());
            });
            return row;
        });
        t.setTooltip(new Tooltip("Double-click an image to open it"));
        t.setPrefHeight(220);
        return t;
    }

    private static TableColumn<Row, String> col(String name, java.util.function.Function<Row, StringProperty> f, double w) {
        var c = new TableColumn<Row, String>(name);
        c.setCellValueFactory(cd -> f.apply(cd.getValue()));
        c.setPrefWidth(w);
        return c;
    }

    private Region logPane() {
        log.setEditable(false);
        log.setWrapText(true);
        log.setPrefRowCount(6);
        log.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        var pane = new TitledPane("Log", log);
        pane.setExpanded(false);
        return pane;
    }

    // ------------------------------------------------------------- backend

    private void checkBackend() {
        var env = new EnvironmentManager();
        if (!env.isReady()) {
            setBackend(Color.web("#d9534f"), "Backend not installed — click Set up…");
            return;
        }
        Thread t = new Thread(() -> {
            try {
                var info = EnvironmentManager.info(env.python(), null);
                boolean cuda = info.has("cuda") && info.get("cuda").getAsBoolean();
                String gpu = cuda && info.has("gpu") ? info.get("gpu").getAsString() : "CPU (slow)";
                String uni2 = SetupWindow.describeUni2(info);
                boolean ok = uni2.contains("ready") || uni2.contains("local file");
                Platform.runLater(() -> setBackend(ok ? Color.web("#2e9e5b") : Color.web("#e0a020"),
                        gpu + " · " + uni2.replace("UNI2-h: ", "UNI2-h ")));
            } catch (Exception e) {
                Platform.runLater(() -> setBackend(Color.web("#d9534f"), "Backend error — click Set up…"));
            }
        }, "tepseg-check");
        t.setDaemon(true);
        t.start();
    }

    private void setBackend(Color c, String text) {
        dot.setFill(c);
        backend.setText(text);
    }

    // ------------------------------------------------------------- targets

    private static Path slidePath(Collection<URI> uris) {
        for (URI uri : uris) {
            if (!"file".equalsIgnoreCase(uri.getScheme()))
                continue;
            try {
                return Path.of(new URI(uri.getScheme(), uri.getSchemeSpecificPart(), null));
            } catch (Exception e) {
                logger.debug("Bad URI {}: {}", uri, e.getMessage());
            }
        }
        return null;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private List<Target> targets() throws IOException {
        Project<?> project = qupath.getProject();
        ImageData<?> current = qupath.getImageData();
        ProjectImageEntry<?> currentEntry = project != null && current != null
                ? ((Project) project).getEntry((ImageData) current) : null;
        List<Target> list = new ArrayList<>();
        if (scopeProject.isSelected()) {
            if (project == null)
                throw new IllegalArgumentException("Open a project first");
            var entries = chosen.isEmpty() ? new ArrayList<ProjectImageEntry<?>>(project.getImageList()) : chosen;
            for (var e : entries) {
                Path p = slidePath(e.getURIs());
                if (p == null) {
                    append("Skipping " + e.getImageName() + " (not a local file)");
                    continue;
                }
                boolean open = currentEntry != null && currentEntry.getID().equals(e.getID());
                list.add(new Target(e.getImageName(), p, open ? current : null, e));
            }
            return list;
        }
        if (current == null)
            throw new IllegalArgumentException("Open an image first");
        Path p = slidePath(current.getServer().getURIs());
        if (p == null)
            throw new IllegalArgumentException("TEPSEG needs a local whole-slide image file");
        String name = currentEntry != null ? currentEntry.getImageName() : current.getServer().getMetadata().getName();
        list.add(new Target(name, p, current, currentEntry));
        return list;
    }

    private Path outputDirGuess() {
        String o = TepsegPrefs.outputDir.get().strip();
        if (!o.isEmpty())
            return Path.of(o);
        Project<?> project = qupath.getProject();
        if (project != null && project.getPath() != null)
            return project.getPath().getParent().resolve("tepseg");
        var data = qupath.getImageData();
        if (data != null) {
            Path p = slidePath(data.getServer().getURIs());
            if (p != null)
                return p.getParent().resolve("tepseg");
        }
        return null;
    }

    // ----------------------------------------------------------------- run

    private void start() {
        var env = new EnvironmentManager();
        if (!env.isReady()) {
            if (Dialogs.showConfirmDialog(NAME, "The Python backend is not installed yet. Set it up now?"))
                new SetupWindow(qupath, this::checkBackend).show();
            return;
        }
        List<Target> targets;
        try {
            targets = targets();
        } catch (Exception e) {
            Dialogs.showErrorMessage(NAME, e.getMessage());
            return;
        }
        if (targets.isEmpty())
            return;
        Path out = outputDirGuess();
        if (out == null)
            out = targets.get(0).slide().getParent().resolve("tepseg");
        lastOutput = out;
        Map<String, Row> bySlide = new HashMap<>();
        rows.clear();
        List<String> cmd = new ArrayList<>(List.of(env.python().toString(), "-u", "-m", "tepseg", "run", "--json",
                "-o", out.toString(), "--stride-um", TepsegPrefs.strideUm.get(),
                "--threshold", String.format(Locale.ROOT, "%.2f", TepsegPrefs.threshold.get()),
                "--min-area-mm2", String.format(Locale.ROOT, "%.2f", TepsegPrefs.minAreaMm2.get())));
        if (TepsegPrefs.skipExisting.get())
            cmd.add("--skip-existing");
        if (!TepsegPrefs.uni2Path.get().isBlank()) {
            cmd.add("--uni2");
            cmd.add(TepsegPrefs.uni2Path.get().strip());
        }
        for (var t : targets) {
            cmd.add(t.slide().toString());
            var row = new Row(t);
            rows.add(row);
            bySlide.put(t.slide().toString(), row);
        }
        run.setDisable(true);
        cancel.setDisable(false);
        bar.setProgress(-1);
        status.setText("Starting…");
        append("Segmenting " + targets.size() + " image(s) → " + out);
        final Path outDir = out;
        Thread worker = new Thread(() -> runProcess(cmd, bySlide, targets.size(), outDir), "tepseg-run");
        worker.setDaemon(true);
        worker.start();
    }

    private void runProcess(List<String> cmd, Map<String, Row> bySlide, int n, Path out) {
        int[] done = {0};
        boolean[] weightsError = {false};
        try {
            var pb = new ProcessBuilder(cmd);
            pb.environment().put("PYTHONIOENCODING", "utf-8");
            process = pb.start();
            Process p = process;
            Thread err = new Thread(() -> {
                try (var r = new BufferedReader(new InputStreamReader(p.getErrorStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = r.readLine()) != null)
                        if (line.contains("ERROR") || line.contains("Traceback") || line.contains("Error") || line.contains("error:"))
                            append(line);
                } catch (IOException ignored) {
                }
            }, "tepseg-stderr");
            err.setDaemon(true);
            err.start();
            try (var r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = r.readLine()) != null) {
                    JsonObject ev;
                    try {
                        ev = JsonParser.parseString(line).getAsJsonObject();
                    } catch (Exception e) {
                        continue;
                    }
                    String type = ev.get("event").getAsString();
                    Row row = ev.has("slide") ? bySlide.get(ev.get("slide").getAsString()) : null;
                    switch (type) {
                        case "status" -> {
                            String m = ev.get("message").getAsString();
                            ui(() -> status.setText(m + "…"));
                        }
                        case "error" -> {
                            weightsError[0] = true;
                            append(ev.get("error").getAsString());
                        }
                        case "slide_start" -> ui(() -> {
                            if (row != null) row.status.set("Reading…");
                            status.setText(String.format("%s (%d of %d)", row != null ? row.image.get() : "", done[0] + 1, n));
                        });
                        case "progress" -> {
                            double f = ev.get("done").getAsDouble() / Math.max(1, ev.get("total").getAsDouble());
                            ui(() -> {
                                if (row != null) row.status.set(String.format("Running %d%%", Math.round(100 * f)));
                                bar.setProgress((done[0] + f) / n);
                            });
                        }
                        case "slide_done" -> {
                            done[0]++;
                            if (row != null) {
                                fill(row, ev.get("tumour_mm2").getAsDouble(), ev.get("tissue_mm2").getAsDouble(),
                                        ev.get("fraction").getAsDouble(), ev.get("regions").getAsInt(), ev.get("seconds").getAsDouble());
                                int k = importRegions(row.target, Path.of(ev.getAsJsonObject("files").get("regions").getAsString()));
                                ui(() -> row.status.set(k >= 0 ? "✓ Done" : "✓ Done (not imported)"));
                            }
                            ui(() -> bar.setProgress((double) done[0] / n));
                        }
                        case "slide_skipped" -> {
                            done[0]++;
                            if (row != null) {
                                String id = ev.get("id").getAsString();
                                reuse(row, out, id);
                            }
                        }
                        case "slide_failed" -> {
                            done[0]++;
                            String msg = ev.get("error").getAsString();
                            append((row != null ? row.image.get() : "") + " failed: " + msg);
                            ui(() -> {
                                if (row != null) row.status.set("✗ Failed");
                            });
                        }
                        case "done" -> ui(() -> status.setText(String.format(Locale.ROOT, "Finished · %d image(s), %d skipped, %d failed · %s",
                                ev.get("slides").getAsInt(), ev.get("skipped").getAsInt(), ev.get("failed").getAsInt(),
                                duration(ev.get("seconds").getAsDouble()))));
                        default -> { }
                    }
                }
            }
            int code = p.waitFor();
            if (weightsError[0])
                ui(() -> {
                    status.setText("UNI2-h is not available");
                    if (Dialogs.showConfirmDialog(NAME, "UNI2-h is not available yet (gated model: request access and "
                            + "sign in, or choose a local copy). Open the setup window?"))
                        new SetupWindow(qupath, this::checkBackend).show();
                });
            else if (code != 0 && done[0] == 0)
                append("TEPSEG stopped (exit code " + code + ")");
        } catch (Exception e) {
            logger.error("TEPSEG run failed", e);
            append("Failed: " + e.getMessage());
        } finally {
            process = null;
            ui(() -> {
                run.setDisable(false);
                cancel.setDisable(true);
                if (bar.getProgress() < 0)
                    bar.setProgress(0);
                rows.forEach(r -> {
                    String s = r.status.get();
                    if (s.equals("Queued") || s.startsWith("Running") || s.startsWith("Reading"))
                        r.status.set("Not run");
                });
            });
        }
    }

    private void fill(Row row, double tumour, double tissue, double fraction, int regions, double seconds) {
        ui(() -> {
            row.tumour.set(String.format(Locale.ROOT, "%.2f", tumour));
            row.tissue.set(String.format(Locale.ROOT, "%.1f", tissue));
            row.fraction.set(String.format(Locale.ROOT, "%.1f", 100 * fraction));
            row.regions.set(Integer.toString(regions));
            row.time.set(seconds >= 0 ? duration(seconds) : "");
        });
    }

    /** A skipped image: show its earlier numbers and import its earlier regions. */
    private void reuse(Row row, Path out, String id) {
        try {
            var js = JsonParser.parseString(Files.readString(out.resolve("slides").resolve(id + ".json"))).getAsJsonObject();
            fill(row, js.get("tumour_mm2").getAsDouble(), js.get("tissue_mm2").getAsDouble(),
                    js.get("tumour_fraction").getAsDouble(), js.get("regions").getAsInt(), -1);
        } catch (Exception e) {
            logger.debug("No earlier summary for {}", id);
        }
        Path regions = out.resolve("regions").resolve(id + ".geojson");
        int k = Files.isRegularFile(regions) ? importRegions(row.target, regions) : -1;
        ui(() -> row.status.set(k >= 0 ? "✓ Earlier result" : "Skipped"));
    }

    private void stop() {
        Process p = process;
        if (p == null)
            return;
        p.descendants().forEach(ProcessHandle::destroyForcibly);   // the slide-reading worker processes
        p.destroyForcibly();
        append("Cancelled");
        status.setText("Cancelled");
    }

    // -------------------------------------------------------------- import

    /** Replace earlier TEPSEG regions of an image with the new ones; returns the number imported, or -1. */
    private int importRegions(Target t, Path file) {
        List<PathObject> objects = new ArrayList<>();
        try {
            for (PathObject o : PathIO.readObjects(file)) {
                var a = PathObjects.createAnnotationObject(o.getROI(), o.getPathClass(), o.getMeasurementList());
                a.setName(NAME);
                objects.add(a);
            }
        } catch (IOException e) {
            append("Could not read " + file + ": " + e.getMessage());
            return -1;
        }
        if (t.open() != null) {
            ui(() -> replace(t.open(), objects));
            return objects.size();
        }
        if (t.entry() != null) {
            try {
                ImageData<?> data = t.entry().readImageData();
                replace(data, objects);
                saveImageData(t.entry(), data);
                return objects.size();
            } catch (Exception e) {
                append("Could not save regions into " + t.name() + ": " + e.getMessage());
            }
        }
        return -1;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private void openEntry(ProjectImageEntry<?> entry) {
        qupath.openImageEntry((ProjectImageEntry) entry);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static void saveImageData(ProjectImageEntry<?> entry, ImageData<?> data) throws IOException {
        ((ProjectImageEntry) entry).saveImageData(data);
    }

    private static void replace(ImageData<?> data, List<PathObject> objects) {
        var h = data.getHierarchy();
        var old = h.getAnnotationObjects().stream()
                .filter(o -> NAME.equals(o.getName()) && o.getPathClass() != null && CLASS.equals(o.getPathClass().toString()))
                .toList();
        if (!old.isEmpty())
            h.removeObjects(old, true);
        h.addObjects(objects);
        h.fireHierarchyChangedEvent(TepsegWindow.class);
    }

    // ---------------------------------------------------------------- misc

    private void openPath(Path p) {
        if (p == null || !Files.exists(p)) {
            Dialogs.showWarningNotification(NAME, "Nothing there yet: run TEPSEG first");
            return;
        }
        Thread t = new Thread(() -> {
            try {
                Desktop.getDesktop().open(p.toFile());
            } catch (Exception e) {
                append("Could not open " + p + ": " + e.getMessage());
            }
        }, "tepseg-open");
        t.setDaemon(true);
        t.start();
    }

    private static String duration(double s) {
        if (s < 60)
            return String.format(Locale.ROOT, "%.0f s", s);
        return String.format(Locale.ROOT, "%d min %02d s", (int) s / 60, (int) s % 60);
    }

    private void append(String s) {
        ui(() -> log.appendText(s + "\n"));
    }

    private static void ui(Runnable r) {
        if (Platform.isFxApplicationThread()) r.run();
        else Platform.runLater(r);
    }
}

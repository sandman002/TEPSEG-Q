package qupath.ext.tepseg.env;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.tepseg.TepsegPrefs;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Consumer;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipInputStream;

/**
 * Creates and maintains a self-contained Python environment for TEPSEG.
 * <p>
 * Everything lives under {@link TepsegPrefs#envDir} (default {@code ~/.tepseg}):
 * a pinned <a href="https://docs.astral.sh/uv/">uv</a> binary, a uv-managed
 * Python 3.11 and a virtual environment with {@code tepseg} installed. uv
 * selects the PyTorch build matching the NVIDIA driver
 * ({@code --torch-backend auto}), so users never pick CUDA versions or
 * install conda. Deleting the folder uninstalls everything.
 */
public class EnvironmentManager {

    private static final Logger logger = LoggerFactory.getLogger(EnvironmentManager.class);

    /** Pinned for reproducible installs; bump deliberately. */
    public static final String UV_VERSION = "0.13.1";
    public static final String PYTHON_VERSION = "3.11";

    private final Path root;

    public EnvironmentManager() {
        this(Path.of(TepsegPrefs.envDir.get()));
    }

    public EnvironmentManager(Path root) {
        this.root = root;
    }

    public Path root() {
        return root;
    }

    private static boolean isWindows() {
        return System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win");
    }

    public Path uvExecutable() {
        return root.resolve("uv").resolve(isWindows() ? "uv.exe" : "uv");
    }

    public Path venv() {
        return root.resolve("env");
    }

    public Path managedPython() {
        return isWindows() ? venv().resolve("Scripts").resolve("python.exe") : venv().resolve("bin").resolve("python");
    }

    /** The Python to run: the user's override if set, otherwise the managed environment. */
    public Path python() {
        String override = TepsegPrefs.pythonOverride.get();
        if (override != null && !override.isBlank())
            return Path.of(override.strip());
        return managedPython();
    }

    public boolean isReady() {
        return Files.isRegularFile(python());
    }

    // ------------------------------------------------------------- install

    /**
     * Install or repair the environment. Blocking; run on a background thread.
     *
     * @param log      receives human-readable progress lines
     * @param step     receives (stepIndex, stepName) as each of the 5 steps starts
     */
    public JsonObject install(Consumer<String> log, StepListener step) throws IOException, InterruptedException {
        Files.createDirectories(root);
        step.onStep(0, "Downloading uv " + UV_VERSION);
        ensureUv(log);

        step.onStep(1, "Installing Python " + PYTHON_VERSION);
        if (!Files.isRegularFile(managedPython()))
            run(log, List.of(uvExecutable().toString(), "venv", "--python", PYTHON_VERSION, "--seed", venv().toString()));
        else
            log.accept("Environment exists: " + venv());

        step.onStep(2, "Installing TEPSEG and PyTorch (GPU build)");
        String spec = TepsegPrefs.packageSpec.get().strip();
        String backend = savedTorchBackend();
        run(log, List.of(uvExecutable().toString(), "pip", "install", "--python", managedPython().toString(),
                "--torch-backend", backend, "--upgrade", spec));
        ensureWorkingTorch(log, backend);

        step.onStep(3, "Downloading UNI2-h (gated model)");
        // not fatal: UNI2-h needs approved access and a Hugging Face sign-in, done in the setup window
        var fetched = tepseg(managedPython(), log, null, "fetch-weights");
        if (fetched.has("ok") && fetched.get("ok").getAsBoolean())
            log.accept("UNI2-h ready: " + fetched.get("path").getAsString());
        else
            log.accept("UNI2-h not available yet: " + (fetched.has("message") ? fetched.get("message").getAsString() : "?"));

        step.onStep(4, "Checking GPU");
        var info = info(managedPython(), log);
        log.accept(describe(info));
        return info;
    }

    // ---------------------------------------------------- torch health check

    /** Newest first. "auto" can pick a CUDA build that is too new for a driver/OS (seen: cu132 on WSL2). */
    private static final List<String> TORCH_FALLBACKS = List.of("cu130", "cu128", "cu126", "cu124");

    private Path torchBackendFile() {
        return root.resolve("torch-backend.txt");
    }

    private String savedTorchBackend() {
        try {
            if (Files.isRegularFile(torchBackendFile()))
                return Files.readString(torchBackendFile()).strip();
        } catch (IOException e) {
            logger.debug("Could not read {}", torchBackendFile());
        }
        return "auto";
    }

    /**
     * Start PyTorch in a separate process and make sure it can use the GPU.
     * If it crashes or cannot see a GPU that nvidia-smi can see, reinstall
     * older CUDA builds until one works (CPU build as a last resort).
     */
    private void ensureWorkingTorch(Consumer<String> log, String installed) throws IOException, InterruptedException {
        boolean nvidia = hasNvidiaGpu();
        TorchStatus st = torchStatus(managedPython());
        log.accept("PyTorch check (" + installed + "): " + st + (nvidia ? " — NVIDIA GPU present" : " — no NVIDIA GPU found"));
        if (st == TorchStatus.GPU || (st == TorchStatus.CPU_ONLY && !nvidia))
            return;
        for (String be : TORCH_FALLBACKS) {
            log.accept("Reinstalling PyTorch with CUDA build " + be + " …");
            run(log, List.of(uvExecutable().toString(), "pip", "install", "--python", managedPython().toString(),
                    "--torch-backend", be, "--reinstall-package", "torch", "--reinstall-package", "torchvision",
                    "torch", "torchvision"));
            st = torchStatus(managedPython());
            log.accept("PyTorch check (" + be + "): " + st);
            if (st == TorchStatus.GPU) {
                Files.writeString(torchBackendFile(), be);
                return;
            }
        }
        log.accept("WARNING: no GPU build of PyTorch works here; installing the CPU build (TEPSEG will be slow).");
        run(log, List.of(uvExecutable().toString(), "pip", "install", "--python", managedPython().toString(),
                "--torch-backend", "cpu", "--reinstall-package", "torch", "--reinstall-package", "torchvision",
                "torch", "torchvision"));
        Files.writeString(torchBackendFile(), "cpu");
    }

    enum TorchStatus { GPU, CPU_ONLY, BROKEN }

    static TorchStatus torchStatus(Path python) throws InterruptedException {
        try {
            var pb = new ProcessBuilder(python.toString(), "-c",
                    "import torch; ok = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()); "
                            + "torch.zeros(1).to('cuda' if torch.cuda.is_available() else 'cpu'); print('GPU' if ok else 'CPU')")
                    .redirectErrorStream(true);
            Process p = pb.start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8).strip();
            if (!p.waitFor(180, java.util.concurrent.TimeUnit.SECONDS)) {
                p.destroyForcibly();
                return TorchStatus.BROKEN;
            }
            if (p.exitValue() != 0)
                return TorchStatus.BROKEN;  // e.g. segfault on CUDA init
            return out.endsWith("GPU") ? TorchStatus.GPU : TorchStatus.CPU_ONLY;
        } catch (IOException e) {
            return TorchStatus.BROKEN;
        }
    }

    static boolean hasNvidiaGpu() {
        try {
            Process p = new ProcessBuilder("nvidia-smi", "-L").redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            return p.waitFor() == 0 && out.contains("GPU");
        } catch (Exception e) {
            return false;
        }
    }

    /** Remove the managed environment (keeps uv and downloaded Python). */
    public void removeEnvironment(Consumer<String> log) throws IOException {
        if (Files.exists(venv())) {
            log.accept("Removing " + venv());
            try (var paths = Files.walk(venv())) {
                paths.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }

    /** Run {@code python -m tepseg info} and parse its JSON. */
    public static JsonObject info(Path python, Consumer<String> log) throws IOException, InterruptedException {
        var pb = new ProcessBuilder(python.toString(), "-m", "tepseg", "info");
        pb.environment().put("PYTHONIOENCODING", "utf-8");
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        String err = new String(p.getErrorStream().readAllBytes(), StandardCharsets.UTF_8);
        if (p.waitFor() != 0) {
            if (log != null)
                log.accept(err);
            throw new IOException("tepseg info failed (is tepseg installed in " + python + "?)");
        }
        int start = out.indexOf('{');
        return JsonParser.parseString(out.substring(Math.max(start, 0))).getAsJsonObject();
    }

    /**
     * Run {@code python -m tepseg <args>} and parse the last JSON object it prints.
     * {@code stdin} (may be null) is written to the process, e.g. a token that must not appear on the command line.
     */
    public static JsonObject tepseg(Path python, Consumer<String> log, String stdin, String... args)
            throws IOException, InterruptedException {
        var cmd = new ArrayList<String>(List.of(python.toString(), "-m", "tepseg"));
        cmd.addAll(List.of(args));
        var pb = new ProcessBuilder(cmd);
        pb.environment().put("PYTHONIOENCODING", "utf-8");
        Process p = pb.start();
        try (var w = p.getOutputStream()) {
            if (stdin != null)
                w.write((stdin + "\n").getBytes(StandardCharsets.UTF_8));
        }
        var errReader = new Thread(() -> {
            try (var r = new BufferedReader(new InputStreamReader(p.getErrorStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = r.readLine()) != null)
                    if (log != null && !line.isBlank())
                        log.accept(line);
            } catch (IOException ignored) {
            }
        });
        errReader.setDaemon(true);
        errReader.start();
        String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        p.waitFor();
        int start = out.lastIndexOf("\n{");
        start = start >= 0 ? start + 1 : out.indexOf('{');
        if (start < 0)
            throw new IOException("tepseg " + args[0] + " printed no result (is tepseg installed in " + python + "?)");
        try {
            return JsonParser.parseString(out.substring(start)).getAsJsonObject();
        } catch (Exception e) {
            int first = out.indexOf('{');
            return JsonParser.parseString(out.substring(Math.max(first, 0))).getAsJsonObject();
        }
    }

    public static String describe(JsonObject info) {
        boolean cuda = info.has("cuda") && info.get("cuda").getAsBoolean();
        String gpu = info.has("gpu") && !info.get("gpu").isJsonNull() ? info.get("gpu").getAsString() : null;
        return String.format("TEPSEG %s · PyTorch %s · %s",
                str(info, "tepseg"), str(info, "torch"),
                cuda ? "GPU: " + gpu + (info.has("gpu_memory_gb") ? " (" + info.get("gpu_memory_gb").getAsString() + " GB)" : "")
                        : (gpu != null ? "GPU: " + gpu : "no GPU found — will run on CPU (slow)"));
    }

    private static String str(JsonObject o, String k) {
        return o.has(k) && !o.get(k).isJsonNull() ? o.get(k).getAsString() : "?";
    }

    // ------------------------------------------------------------------ uv

    private void ensureUv(Consumer<String> log) throws IOException, InterruptedException {
        Path uv = uvExecutable();
        if (Files.isRegularFile(uv)) {
            log.accept("uv already present: " + uv);
            return;
        }
        String asset = uvAsset();
        String url = "https://github.com/astral-sh/uv/releases/download/" + UV_VERSION + "/" + asset;
        log.accept("Downloading " + url);
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(30)).build();
        var resp = client.send(HttpRequest.newBuilder(URI.create(url)).GET().build(), HttpResponse.BodyHandlers.ofInputStream());
        if (resp.statusCode() != 200)
            throw new IOException("Download failed (HTTP " + resp.statusCode() + "): " + url);
        Files.createDirectories(uv.getParent());
        try (InputStream in = new BufferedInputStream(resp.body())) {
            if (asset.endsWith(".zip"))
                extractFromZip(in, uv.getFileName().toString(), uv);
            else
                extractFromTarGz(in, uv.getFileName().toString(), uv);
        }
        if (!uv.toFile().setExecutable(true))
            logger.debug("Could not mark {} executable", uv);
        log.accept("uv installed: " + uv);
    }

    static String uvAsset() {
        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        String cpu = (arch.contains("aarch64") || arch.contains("arm64")) ? "aarch64" : "x86_64";
        if (os.contains("win"))
            return "uv-" + cpu + "-pc-windows-msvc.zip";
        if (os.contains("mac") || os.contains("darwin"))
            return "uv-" + cpu + "-apple-darwin.tar.gz";
        return "uv-" + cpu + "-unknown-linux-gnu.tar.gz";
    }

    private static void extractFromZip(InputStream in, String name, Path dest) throws IOException {
        try (var zip = new ZipInputStream(in)) {
            for (var e = zip.getNextEntry(); e != null; e = zip.getNextEntry()) {
                if (!e.isDirectory() && Path.of(e.getName()).getFileName().toString().equals(name)) {
                    Files.copy(zip, dest, StandardCopyOption.REPLACE_EXISTING);
                    return;
                }
            }
        }
        throw new IOException(name + " not found in uv archive");
    }

    /** Minimal ustar reader: enough to pull one file out of uv's release tarball. */
    private static void extractFromTarGz(InputStream in, String name, Path dest) throws IOException {
        try (var gz = new GZIPInputStream(in)) {
            byte[] header = new byte[512];
            while (gz.readNBytes(header, 0, 512) == 512) {
                String entry = new String(header, 0, 100, StandardCharsets.US_ASCII).trim().replace("\0", "");
                if (entry.isEmpty())
                    break;
                String sizeField = new String(header, 124, 12, StandardCharsets.US_ASCII).replace("\0", "").trim();
                long size = sizeField.isEmpty() ? 0 : Long.parseLong(sizeField, 8);
                byte type = header[156];
                long padded = (size + 511) / 512 * 512;
                if ((type == '0' || type == 0) && Path.of(entry).getFileName().toString().equals(name)) {
                    byte[] data = gz.readNBytes((int) size);
                    Files.write(dest, data);
                    return;
                }
                gz.skipNBytes(padded);
            }
        }
        throw new IOException(name + " not found in uv archive");
    }

    // ----------------------------------------------------------- processes

    private void run(Consumer<String> log, List<String> command) throws IOException, InterruptedException {
        log.accept("$ " + String.join(" ", command));
        var pb = new ProcessBuilder(command).redirectErrorStream(true);
        Map<String, String> env = pb.environment();
        env.put("UV_PYTHON_INSTALL_DIR", root.resolve("python").toString());  // keep Python inside our folder
        env.put("UV_CACHE_DIR", root.resolve("cache").toString());
        env.put("PYTHONIOENCODING", "utf-8");
        env.put("UV_NO_PROGRESS", "1");
        Process p = pb.start();
        try (var reader = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null)
                log.accept(line);
        }
        int code = p.waitFor();
        if (code != 0)
            throw new IOException("Command failed (exit " + code + "): " + command.get(0) + " " + command.get(1));
    }

    @FunctionalInterface
    public interface StepListener {
        void onStep(int index, String name);
    }
}

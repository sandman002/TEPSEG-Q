package qupath.ext.tepseg.ui;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.control.ListView;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.CheckBoxListCell;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Window;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

import java.util.Collection;
import java.util.List;
import java.util.Locale;

/** Checklist dialog for picking project images. */
final class ImageChooser {

    private ImageChooser() {}

    private record Item(ProjectImageEntry<?> entry, BooleanProperty selected) {
        @Override
        public String toString() {
            return entry.getImageName();
        }
    }

    static List<ProjectImageEntry<?>> choose(Window owner, Project<?> project, Collection<ProjectImageEntry<?>> current) {
        var ids = current.stream().map(ProjectImageEntry::getID).toList();
        var items = FXCollections.observableArrayList(project.getImageList().stream()
                .map(e -> new Item(e, new SimpleBooleanProperty(current.isEmpty() || ids.contains(e.getID()))))
                .toList());
        var filtered = items.filtered(i -> true);
        var list = new ListView<>(filtered);
        list.setCellFactory(CheckBoxListCell.forListView(Item::selected));
        list.setPrefSize(480, 420);

        var filter = new TextField();
        filter.setPromptText("Filter by name");
        filter.textProperty().addListener((o, a, t) -> filtered.setPredicate(
                i -> t == null || t.isBlank() || i.toString().toLowerCase(Locale.ROOT).contains(t.toLowerCase(Locale.ROOT))));
        var all = new Button("Select all");
        all.setOnAction(e -> filtered.forEach(i -> i.selected().set(true)));
        var none = new Button("Select none");
        none.setOnAction(e -> filtered.forEach(i -> i.selected().set(false)));

        var content = new VBox(8, new HBox(8, filter, all, none), list);
        content.setPadding(new Insets(10));
        var dialog = new Dialog<ButtonType>();
        dialog.initOwner(owner);
        dialog.setTitle("TEPSEG – choose images");
        dialog.getDialogPane().setContent(content);
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
        var result = dialog.showAndWait();
        if (result.isEmpty() || result.get() != ButtonType.OK)
            return null;
        return items.stream().filter(i -> i.selected().get()).<ProjectImageEntry<?>>map(Item::entry).toList();
    }
}

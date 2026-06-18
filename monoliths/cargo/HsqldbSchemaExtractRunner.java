package se.citerus.dddsample;

import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import javax.sql.DataSource;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.*;
import java.util.*;

@Component
public class HsqldbSchemaExtractRunner implements CommandLineRunner {

    private final DataSource dataSource;

    public HsqldbSchemaExtractRunner(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    @Override
    public void run(String... args) throws Exception {

        Path out = Path.of("data.sql");

        try (Connection conn = dataSource.getConnection();
             Statement stmt = conn.createStatement();
             var writer = Files.newBufferedWriter(out)) {

            Map<String, List<Column>> tableColumns = new LinkedHashMap<>();
            Map<String, List<String>> primaryKeys = new HashMap<>();
            Map<String, List<ForeignKey>> foreignKeys = new HashMap<>();

            // ------------------------------------------------------------
            // Columns
            // ------------------------------------------------------------
            ResultSet cols = stmt.executeQuery("""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE,
                       CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION,
                       IS_NULLABLE, ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'PUBLIC'
                ORDER BY TABLE_NAME, ORDINAL_POSITION
            """);

            while (cols.next()) {
                String table = cols.getString("TABLE_NAME").toUpperCase();
                tableColumns
                        .computeIfAbsent(table, k -> new ArrayList<>())
                        .add(new Column(
                                cols.getString("COLUMN_NAME").toUpperCase(),
                                cols.getString("DATA_TYPE"),
                                cols.getInt("CHARACTER_MAXIMUM_LENGTH"),
                                cols.getInt("NUMERIC_PRECISION"),
                                "NO".equals(cols.getString("IS_NULLABLE"))
                        ));
            }

            // ------------------------------------------------------------
            // Primary keys
            // ------------------------------------------------------------
            ResultSet pks = stmt.executeQuery("""
                SELECT tc.TABLE_NAME, kc.COLUMN_NAME, kc.ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kc
                  ON tc.CONSTRAINT_NAME = kc.CONSTRAINT_NAME
                WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                  AND tc.TABLE_SCHEMA = 'PUBLIC'
                ORDER BY tc.TABLE_NAME, kc.ORDINAL_POSITION
            """);

            while (pks.next()) {
                primaryKeys
                        .computeIfAbsent(pks.getString("TABLE_NAME").toUpperCase(),
                                k -> new ArrayList<>())
                        .add(pks.getString("COLUMN_NAME").toUpperCase());
            }

            // ------------------------------------------------------------
            // Foreign keys
            // ------------------------------------------------------------
            ResultSet fks = stmt.executeQuery("""
                SELECT fk.TABLE_NAME  AS fk_table,
                       fk.COLUMN_NAME AS fk_column,
                       pk.TABLE_NAME  AS pk_table,
                       pk.COLUMN_NAME AS pk_column,
                       fk.ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE fk
                  ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pk
                  ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
                WHERE fk.TABLE_SCHEMA = 'PUBLIC'
                ORDER BY fk.TABLE_NAME, fk.ORDINAL_POSITION
            """);

            while (fks.next()) {
                foreignKeys
                        .computeIfAbsent(fks.getString("fk_table").toUpperCase(),
                                k -> new ArrayList<>())
                        .add(new ForeignKey(
                                fks.getString("fk_column").toUpperCase(),
                                fks.getString("pk_table").toUpperCase(),
                                fks.getString("pk_column").toUpperCase()
                        ));
            }

            // ------------------------------------------------------------
            // Emit CREATE TABLE statements (parser-safe)
            // ------------------------------------------------------------
            for (String table : tableColumns.keySet()) {

                writer.write("CREATE TABLE " + table + " (\n");
                List<String> lines = new ArrayList<>();

                for (Column c : tableColumns.get(table)) {
                    String type = toMySqlType(c);
                    lines.add("  " + c.name + " " + type +
                            (c.notNull ? " NOT NULL" : ""));
                }

                if (primaryKeys.containsKey(table)) {
                    lines.add("  PRIMARY KEY (" +
                            String.join(", ", primaryKeys.get(table)) + ")");
                }

                if (foreignKeys.containsKey(table)) {
                    for (ForeignKey fk : foreignKeys.get(table)) {
                        lines.add("  FOREIGN KEY (" + fk.column + ") REFERENCES " +
                                fk.refTable + "(" + fk.refColumn + ")");
                    }
                }

                writer.write(String.join(",\n", lines));
                writer.write("\n);\n\n"); // <-- IMPORTANT: no ENGINE here
            }
        }

        System.out.println("Schema extracted to data.sql (parser compatible)");
    }

    // ------------------------------------------------------------
    // MySQL type normalization
    // ------------------------------------------------------------
    private static String toMySqlType(Column c) {
        String t = c.type.toUpperCase();

        return switch (t) {
            case "CHARACTER VARYING", "VARCHAR" ->
                    "VARCHAR(" + c.charLen + ")";
            case "CHARACTER", "CHAR" ->
                    "CHAR(" + c.charLen + ")";
            case "BIGINT" -> "BIGINT";
            case "INTEGER" -> "INT";
            case "SMALLINT" -> "SMALLINT";
            case "BOOLEAN" -> "BOOLEAN";
            case "DECIMAL", "NUMERIC" ->
                    "DECIMAL(" + c.numPrecision + ")";
            case "DOUBLE PRECISION" -> "DOUBLE";
            case "REAL" -> "FLOAT";
            case "TIMESTAMP" -> "DATETIME";
            case "DATE" -> "DATE";
            case "TIME" -> "TIME";
            default -> t;
        };
    }

    // ------------------------------------------------------------
    // Helper classes
    // ------------------------------------------------------------
    static class Column {
        String name;
        String type;
        int charLen;
        int numPrecision;
        boolean notNull;

        Column(String name, String type, int charLen, int numPrecision, boolean notNull) {
            this.name = name;
            this.type = type;
            this.charLen = charLen;
            this.numPrecision = numPrecision;
            this.notNull = notNull;
        }
    }

    static class ForeignKey {
        String column;
        String refTable;
        String refColumn;

        ForeignKey(String column, String refTable, String refColumn) {
            this.column = column;
            this.refTable = refTable;
            this.refColumn = refColumn;
        }
    }
}

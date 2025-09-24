package za.co.wethinkcode.utilities;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import org.json.JSONArray;
import org.json.JSONObject;
import java.util.Arrays;

public class TextVectorizer {
    private static final String OLLAMA_API_URL = "http://localhost:11434/api/embeddings";
    private static final String DB_URL = "jdbc:sqlite:/Users/denzhedzebu/Desktop/RAGNARIZE/embeddings_vss.db";
    private Connection conn;

    public TextVectorizer() {
        initializeDatabase();
    }

    // Initialize SQLite database with vss extensions
    private void initializeDatabase() {
        try {
            // Explicitly load dylibs
            System.load("/Users/denzhedzebu/Desktop/RAGNARIZE/libs/vector0.dylib");
            System.load("/Users/denzhedzebu/Desktop/RAGNARIZE/libs/vss0.dylib");

            conn = DriverManager.getConnection(DB_URL);
            Statement stmt = conn.createStatement();

            // Create metadata table
            String metadataSql = "CREATE TABLE IF NOT EXISTS embeddings_metadata (" +
                    "id INTEGER PRIMARY KEY AUTOINCREMENT," +
                    "input_text TEXT NOT NULL," +
                    "vector_id TEXT NOT NULL UNIQUE," +
                    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)";
            stmt.execute(metadataSql);

            // Create vss table
            String vssSql = "CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_vectors USING vss0(" +
                    "embeddings_metadata(vector_id), " +
                    "embedding BLOB, " +
                    "distance REAL)";
            stmt.execute(vssSql);

            stmt.close();
        } catch (SQLException e) {
            System.err.println("Database initialization error: " + e.getMessage());
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load dylibs: " + e.getMessage());
        }
    }

    // Generate embedding using Ollama API
    public float[] getEmbedding(String inputText) throws Exception {
        URL url = new URL(OLLAMA_API_URL);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("POST");
        connection.setRequestProperty("Content-Type", "application/json");
        connection.setDoOutput(true);

        JSONObject payload = new JSONObject();
        payload.put("model", "nomic-embed-text");
        payload.put("prompt", inputText);

        try (OutputStream os = connection.getOutputStream()) {
            byte[] input = payload.toString().getBytes("utf-8");
            os.write(input, 0, input.length);
        }

        StringBuilder response = new StringBuilder();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(connection.getInputStream(), "utf-8"))) {
            String responseLine;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
        }

        JSONObject jsonResponse = new JSONObject(response.toString());
        JSONArray embeddingArray = jsonResponse.getJSONArray("embedding");
        float[] embedding = new float[embeddingArray.length()];
        for (int i = 0; i < embeddingArray.length(); i++) {
            embedding[i] = (float) embeddingArray.getDouble(i);
        }

        connection.disconnect();
        return embedding;
    }

    // Store embedding in vss table
    public void storeEmbedding(String inputText, float[] embedding) {
        String vectorId = "vec_" + System.currentTimeMillis();
        byte[] embeddingBlob = toBlob(embedding);

        String metadataSql = "INSERT OR IGNORE INTO embeddings_metadata (input_text, vector_id) VALUES (?, ?)";
        String vssSql = "INSERT OR REPLACE INTO embeddings_vectors (vector_id, embedding) VALUES (?, ?)";

        try (PreparedStatement metaPstmt = conn.prepareStatement(metadataSql);
             PreparedStatement vssPstmt = conn.prepareStatement(vssSql)) {

            conn.setAutoCommit(false);

            metaPstmt.setString(1, inputText);
            metaPstmt.setString(2, vectorId);
            metaPstmt.executeUpdate();

            vssPstmt.setString(1, vectorId);
            vssPstmt.setBytes(2, embeddingBlob);
            vssPstmt.executeUpdate();

            conn.commit();
        } catch (SQLException e) {
            try {
                conn.rollback();
            } catch (SQLException rollbackEx) {
                System.err.println("Rollback error: " + rollbackEx.getMessage());
            }
            System.err.println("Error storing embedding: " + e.getMessage());
        } finally {
            try {
                conn.setAutoCommit(true);
            } catch (SQLException e) {
                System.err.println("Auto-commit error: " + e.getMessage());
            }
        }
    }

    // Search top-k similar embeddings using vss
    public String[] searchSimilar(float[] queryEmbedding, int topK) {
        byte[] queryBlob = toBlob(queryEmbedding);
        String[] results = new String[topK];

        String sql = "SELECT rowid, distance FROM embeddings_vectors " +
                "WHERE vss_search(embedding, ?) " +
                "ORDER BY distance " +
                "LIMIT ?";

        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setBytes(1, queryBlob);
            pstmt.setInt(2, topK);

            try (ResultSet rs = pstmt.executeQuery()) {
                int index = 0;
                while (rs.next() && index < topK) {
                    long rowid = rs.getLong("rowid");
                    double distance = rs.getDouble("distance");

                    String textSql = "SELECT input_text FROM embeddings_metadata WHERE id = ?";
                    try (PreparedStatement textPstmt = conn.prepareStatement(textSql)) {
                        textPstmt.setLong(1, rowid);
                        try (ResultSet textRs = textPstmt.executeQuery()) {
                            if (textRs.next()) {
                                results[index] = textRs.getString("input_text") + " (distance: " + distance + ")";
                                index++;
                            }
                        }
                    }
                }
            }
        } catch (SQLException e) {
            System.err.println("Error searching embeddings: " + e.getMessage());
            return new String[0];
        }

        return results;
    }

    // Convert float array to byte blob (little-endian)
    private byte[] toBlob(float[] array) {
        byte[] blob = new byte[array.length * 4];
        for (int i = 0; i < array.length; i++) {
            int bits = Float.floatToIntBits(array[i]);
            blob[i * 4] = (byte) (bits & 0xff);
            blob[i * 4 + 1] = (byte) ((bits >> 8) & 0xff);
            blob[i * 4 + 2] = (byte) ((bits >> 16) & 0xff);
            blob[i * 4 + 3] = (byte) ((bits >> 24) & 0xff);
        }
        return blob;
    }

    // Close database connection
    public void close() {
        try {
            if (conn != null && !conn.isClosed()) {
                conn.close();
            }
        } catch (SQLException e) {
            System.err.println("Error closing database connection: " + e.getMessage());
        }
    }

    // Example usage
    public static void main(String[] args) {
        TextVectorizer vectorizer = new TextVectorizer();
        try {
            String input = "This is a sample text for vectorization";
            float[] embedding = vectorizer.getEmbedding(input);
            vectorizer.storeEmbedding(input, embedding);

            String input2 = "Another sample text for testing";
            float[] embedding2 = vectorizer.getEmbedding(input2);
            vectorizer.storeEmbedding(input2, embedding2);

            String[] similarTexts = vectorizer.searchSimilar(embedding, 5);
            System.out.println("Top 5 similar texts: " + Arrays.toString(similarTexts));
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        } finally {
            vectorizer.close();
        }
    }
}
package za.co.wethinkcode;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.data.message.UserMessage;

import java.util.List;
import java.util.Scanner;

public class RagTool {

    private static final String OLLAMA_BASE_URL = "http://localhost:11434";
    private static final String EMBEDDING_MODEL_NAME = "deepseek-r1:7b";
    private static final String GENERATION_MODEL_NAME = "deepseek-r1:7b";
    private static final int MAX_RETRIEVED_SEGMENTS = 5;
    private static final double MIN_RELEVANCE_SCORE = 0.5;

    // PostgreSQL connection details
    private static final String PG_HOST = "localhost";
    private static final int PG_PORT = 5432;
    private static final String PG_DATABASE = "ragdb";
    private static final String PG_USER = "postgres";
    private static final String PG_PASSWORD = "password";

    public static void main(String[] args) {
        // Initialize models
        EmbeddingModel embeddingModel = OllamaEmbeddingModel.builder()
                .baseUrl(OLLAMA_BASE_URL)
                .modelName(EMBEDDING_MODEL_NAME)
                .build();

        OllamaChatModel chatModel = OllamaChatModel.builder()
                .baseUrl(OLLAMA_BASE_URL)
                .modelName(GENERATION_MODEL_NAME)
                .build();

        // Initialize PostgreSQL vector store
        EmbeddingStore<TextSegment> embeddingStore = PgVectorEmbeddingStore.builder()
                .host(PG_HOST)
                .port(PG_PORT)
                .database(PG_DATABASE)
                .user(PG_USER)
                .password(PG_PASSWORD)
                .table("embeddings")
                .dimension(3584) // For deepseek-r1:7b
                .build();

        // Document splitter for chunking large text (~1000 chars per chunk with overlap)
        DocumentSplitter splitter = DocumentSplitters.recursive(1000, 200);

        Scanner scanner = new Scanner(System.in);

        System.out.println("RAG Tool: Enter 'ingest' to add text, 'query' to ask a question, or 'exit' to quit.");

        while (true) {
            System.out.print("> ");
            String command = scanner.nextLine().trim();

            if (command.equalsIgnoreCase("exit")) {
                break;
            } else if (command.equalsIgnoreCase("ingest")) {
                System.out.println("Enter the text to ingest (end with EOF or blank line):");
                StringBuilder textBuilder = new StringBuilder();
                String line;
                while (!(line = scanner.nextLine()).isBlank()) {
                    textBuilder.append(line).append("\n");
                }
                String text = textBuilder.toString().trim();
                if (!text.isEmpty()) {
                    ingestText(text, embeddingModel, embeddingStore, splitter);
                    System.out.println("Text ingested and vectorized.");
                }
            } else if (command.equalsIgnoreCase("query")) {
                System.out.print("Enter your query: ");
                String query = scanner.nextLine().trim();
                if (!query.isEmpty()) {
                    String response = queryRag(query, embeddingModel, embeddingStore, chatModel);
                    System.out.println("Response: " + response);
                }
            } else {
                System.out.println("Unknown command. Try 'ingest', 'query', or 'exit'.");
            }
        }
    }

    private static void ingestText(String text, EmbeddingModel embeddingModel,
                                   EmbeddingStore<TextSegment> embeddingStore,
                                   DocumentSplitter splitter) {
        Document document = Document.from(text);
        List<TextSegment> segments = splitter.split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        embeddingStore.addAll(embeddings, segments);
    }

    private static String queryRag(String query, EmbeddingModel embeddingModel,
                                   EmbeddingStore<TextSegment> embeddingStore,
                                   OllamaChatModel chatModel) {
        Embedding queryEmbedding = embeddingModel.embed(query).content();
        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(MAX_RETRIEVED_SEGMENTS)
                .minScore(MIN_RELEVANCE_SCORE)
                .build();
        EmbeddingSearchResult<TextSegment> result = embeddingStore.search(searchRequest);
        List<EmbeddingMatch<TextSegment>> relevantMatches = result.matches();

        System.out.println("Found " + relevantMatches.size() + " relevant matches:");
        for (EmbeddingMatch<TextSegment> match : relevantMatches) {
            System.out.println("Score: " + match.score() + ", Text: " + match.embedded().text());
        }

        StringBuilder context = new StringBuilder();
        for (EmbeddingMatch<TextSegment> match : relevantMatches) {
            context.append(match.embedded().text()).append("\n\n");
        }

        if (context.isEmpty()) {
            return chatModel.chat(new UserMessage(query)).aiMessage().text();
        }

        String augmentedPrompt = "You must answer the question based solely on the following context. Do not use any prior knowledge or external information. If the context is insufficient, say 'I don’t have enough information.' Question: " + query + "\n\nContext:\n" + context;
        System.out.println("Augmented Prompt: " + augmentedPrompt); // Debug output
        return chatModel.chat(new UserMessage(augmentedPrompt)).aiMessage().text();
    }
}
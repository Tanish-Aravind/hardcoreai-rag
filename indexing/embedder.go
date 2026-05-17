package indexing

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

const EmbeddingDimension = 768 // Gemini uses 768, not 1536

type Embedder struct {
	apiKey string
	mock   bool
}

func NewMockEmbedder() *Embedder {
	fmt.Println("⚠️  Using MOCK embedder - replace with real API later")
	return &Embedder{mock: true}
}

func NewGeminiEmbedder(apiKey string) *Embedder {
	return &Embedder{apiKey: apiKey, mock: false}
}

func (e *Embedder) EmbedText(text string) ([]float32, error) {
	if e.mock {
		return mockEmbedding(), nil
	}
	return e.geminiEmbed(text)
}

func (e *Embedder) EmbedBatch(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := e.embedWithRetry(text, 3)
		if err != nil {
			return nil, fmt.Errorf("failed to embed text %d: %w", i, err)
		}
		embeddings[i] = emb

		// Progress every 50 chunks
		if (i+1)%50 == 0 {
			fmt.Printf("   embedded %d/%d chunks...\n", i+1, len(texts))
		}

		// Rate limit: wait 1 second between calls
		time.Sleep(1 * time.Second)
	}
	fmt.Printf("✅ Generated %d embeddings (dim=%d)\n", len(embeddings), EmbeddingDimension)
	return embeddings, nil
}

func (e *Embedder) embedWithRetry(text string, maxRetries int) ([]float32, error) {
	for attempt := 0; attempt < maxRetries; attempt++ {
		emb, err := e.geminiEmbed(text)
		if err == nil {
			return emb, nil
		}

		// If rate limited, wait longer and retry
		if strings.Contains(err.Error(), "429") {
			waitTime := time.Duration(30*(attempt+1)) * time.Second
			fmt.Printf("   ⚠️  Rate limited, waiting %v before retry %d/%d...\n", waitTime, attempt+1, maxRetries)
			time.Sleep(waitTime)
			continue
		}

		// Other errors, don't retry
		return nil, err
	}
	return nil, fmt.Errorf("failed after %d retries", maxRetries)
}
func (e *Embedder) geminiEmbed(text string) ([]float32, error) {
	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=%s",
		e.apiKey,
	)

	// Truncate text if too long (Gemini has token limits)
	if len(text) > 8000 {
		text = text[:8000]
	}

	reqBody := map[string]interface{}{
		"model": "models/gemini-embedding-001",
		"content": map[string]interface{}{
			"parts": []map[string]interface{}{
				{"text": text},
			},
		},
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := http.Post(url, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Gemini API error %d: %s", resp.StatusCode, string(respBytes))
	}

	var result struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}

	if err := json.Unmarshal(respBytes, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return result.Embedding.Values, nil
}

func mockEmbedding() []float32 {
	vec := make([]float32, EmbeddingDimension)
	var sum float32
	for i := range vec {
		vec[i] = float32(i%10) * 0.1
		sum += vec[i] * vec[i]
	}
	magnitude := float32(math.Sqrt(float64(sum)))
	for i := range vec {
		vec[i] /= magnitude
	}
	return vec
}

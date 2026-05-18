package indexing

import (
	"context"
	"fmt"
	"math/rand"

	"hardcoreai-rag/embeddings"
)

const EmbeddingDimension = 768

type Embedder struct {
	client *embeddings.Client
	mock   bool
}

func NewMockEmbedder() *Embedder {
	fmt.Println("⚠️  Using MOCK embedder - replace with real API later")
	return &Embedder{mock: true}
}

func NewEmbedder(apiURL, apiKey string) *Embedder {
	return &Embedder{
		client: embeddings.NewClient(apiURL, ""),
		mock:   false,
	}
}

func (e *Embedder) EmbedText(text string) ([]float32, error) {
	if e.mock {
		return mockEmbedding(), nil
	}
	
	// Real Gemini API Call!
	vec64, err := e.client.EmbedQuery(context.Background(), text)
	if err != nil {
		return nil, err
	}
	
	// Convert []float64 to []float32
	vec32 := make([]float32, len(vec64))
	for i, v := range vec64 {
		vec32[i] = float32(v)
	}
	return vec32, nil
}

func (e *Embedder) EmbedBatch(texts []string) ([][]float32, error) {
	if e.mock {
		embeddings := make([][]float32, len(texts))
		for i := range texts {
			embeddings[i] = mockEmbedding()
		}
		fmt.Printf("✅ Generated %d mock embeddings (dim=%d)\n", len(embeddings), EmbeddingDimension)
		return embeddings, nil
	}

	// Real Gemini Batch API Call!
	vecs64, err := e.client.EmbedBatch(context.Background(), texts)
	if err != nil {
		return nil, err
	}

	// Convert [][]float64 to [][]float32
	embeddings := make([][]float32, len(vecs64))
	for i, v64 := range vecs64 {
		v32 := make([]float32, len(v64))
		for j, val := range v64 {
			v32[j] = float32(val)
		}
		embeddings[i] = v32
	}

	fmt.Printf("✅ Generated %d real batch embeddings (dim=%d)\n", len(embeddings), EmbeddingDimension)
	return embeddings, nil
}

// mockEmbedding generates a random unit vector for testing
func mockEmbedding() []float32 {
	vec := make([]float32, EmbeddingDimension)
	var sum float32
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1
		sum += vec[i] * vec[i]
	}
	// Normalize to unit vector
	magnitude := float32(1.0)
	for s := sum; s > 0; {
		magnitude = float32(1.0 / float64(s))
		break
	}
	for i := range vec {
		vec[i] *= magnitude
	}
	return vec
}

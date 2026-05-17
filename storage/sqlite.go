package storage

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"

	_ "github.com/ncruces/go-sqlite3/driver"
	_ "github.com/ncruces/go-sqlite3/embed"
)

type DB struct {
	conn *sql.DB
}

func NewDB(dbPath string) (*DB, error) {
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create db directory: %w", err)
	}

	conn, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if err := conn.Ping(); err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	db := &DB{conn: conn}

	if err := db.createTables(); err != nil {
		return nil, fmt.Errorf("failed to create tables: %w", err)
	}

	fmt.Println("✅ Database connected and tables ready")
	return db, nil
}

func (db *DB) createTables() error {
	schema := `
	CREATE TABLE IF NOT EXISTS documents (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		mongo_id TEXT UNIQUE,
		filename TEXT NOT NULL,
		local_path TEXT,
		source_url TEXT,
		doc_type TEXT,
		chip_family TEXT,
		chip_model TEXT,
		version TEXT,
		processing_status TEXT DEFAULT 'pending',
		error_message TEXT,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS chunks (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		document_id INTEGER NOT NULL,
		chunk_text TEXT NOT NULL,
		section_title TEXT,
		subsection_title TEXT,
		peripheral TEXT,
		register_name TEXT,
		page_number INTEGER,
		token_count INTEGER,
		chunk_index INTEGER,
		metadata TEXT,
		embedding BLOB,
		FOREIGN KEY(document_id) REFERENCES documents(id)
	);

	CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
		chunk_text,
		section_title,
		peripheral,
		register_name,
		content='chunks',
		content_rowid='id'
	);

	CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
		INSERT INTO chunk_fts(rowid, chunk_text, section_title, peripheral, register_name)
		VALUES (new.id, new.chunk_text, new.section_title, new.peripheral, new.register_name);
	END;

	CREATE INDEX IF NOT EXISTS idx_documents_chip_family ON documents(chip_family);
	CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
	CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
	CREATE INDEX IF NOT EXISTS idx_chunks_peripheral ON chunks(peripheral);
	CREATE INDEX IF NOT EXISTS idx_chunks_register ON chunks(register_name);
	`

	_, err := db.conn.Exec(schema)
	return err
}

func (db *DB) InsertDocument(doc Document) (int64, error) {
	result, err := db.conn.Exec(`
		INSERT INTO documents 
			(mongo_id, filename, local_path, source_url, doc_type, chip_family, chip_model, version, processing_status)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		doc.MongoID, doc.Filename, doc.LocalPath, doc.SourceURL,
		doc.DocType, doc.ChipFamily, doc.ChipModel, doc.Version, "processing",
	)
	if err != nil {
		return 0, fmt.Errorf("failed to insert document: %w", err)
	}

	id, err := result.LastInsertId()
	if err != nil {
		return 0, err
	}

	fmt.Printf("📄 Inserted document: %s (id=%d)\n", doc.Filename, id)
	return id, nil
}

func (db *DB) InsertChunksAndReturnIDs(chunks []Chunk) ([]int, error) {
	tx, err := db.conn.Begin()
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`
		INSERT INTO chunks
			(document_id, chunk_text, section_title, subsection_title, peripheral, register_name, page_number, token_count, chunk_index, metadata)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	ids := make([]int, len(chunks))
	for i, chunk := range chunks {
		result, err := stmt.Exec(
			chunk.DocumentID, chunk.ChunkText, chunk.SectionTitle, chunk.SubsectionTitle,
			chunk.Peripheral, chunk.RegisterName, chunk.PageNumber, chunk.TokenCount,
			chunk.ChunkIndex, chunk.Metadata,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to insert chunk %d: %w", chunk.ChunkIndex, err)
		}
		id, _ := result.LastInsertId()
		ids[i] = int(id)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("failed to commit: %w", err)
	}

	fmt.Printf("✅ Inserted %d chunks into database\n", len(chunks))
	return ids, nil
}

func (db *DB) UpdateDocumentStatus(id int64, status, errMsg string) error {
	_, err := db.conn.Exec(`
		UPDATE documents SET processing_status = ?, error_message = ? WHERE id = ?`,
		status, errMsg, id,
	)
	return err
}

func (db *DB) GetChunksByPeripheral(peripheral string) ([]Chunk, error) {
	rows, err := db.conn.Query(`
		SELECT id, document_id, chunk_text, section_title, peripheral, register_name, page_number, token_count, chunk_index
		FROM chunks WHERE peripheral = ?`, peripheral)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var chunks []Chunk
	for rows.Next() {
		var c Chunk
		err := rows.Scan(
			&c.ID, &c.DocumentID, &c.ChunkText, &c.SectionTitle,
			&c.Peripheral, &c.RegisterName, &c.PageNumber, &c.TokenCount, &c.ChunkIndex,
		)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, c)
	}
	return chunks, nil
}

func (db *DB) Close() error {
	return db.conn.Close()
}

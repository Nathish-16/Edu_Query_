import React, { useRef, useState } from 'react';
import { UploadCloud, BookOpen, Presentation, FileText, Trash2, Activity, RefreshCw } from 'lucide-react';
import { apiFetch, parseApiError, readJson } from '../api';

export default function Sidebar({
  documents,
  status,
  onDeleteDoc,
  onRefreshLibrary,
  onUploadStart,
  onUploadSuccess,
  onUploadError,
}) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');

  const getDocIcon = (type) => {
    switch (type?.toLowerCase()) {
      case 'pdf':
        return <BookOpen size={16} className="doc-icon" />;
      case 'pptx':
      case 'ppt':
        return <Presentation size={16} className="doc-icon" />;
      default:
        return <FileText size={16} className="doc-icon" />;
    }
  };

  const handleFileChange = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    await uploadFiles(files);
  };

  const readError = async (response) => {
    const payload = await readJson(response);
    return parseApiError(payload, 'Server error');
  };

  const uploadFiles = async (files) => {
    onUploadStart();
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      setUploadProgress(`Uploading ${i + 1}/${files.length}...`);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await apiFetch('/api/upload', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data = await readJson(response);
          const fileType = file.name.split('.').pop()?.toLowerCase() || 'unknown';
          const indexedNote = data.chunks_count
            ? `Successfully indexed ${file.name} (${data.chunks_count} chunks)`
            : `Uploaded ${file.name}. It is in your library while indexing finishes.`;
          await onUploadSuccess(
            indexedNote,
            { source: data.filename || file.name, type: fileType },
          );
        } else {
          const errDetail = await readError(response);
          onUploadError(`Failed to process ${file.name}: ${errDetail}`);
        }
      } catch (err) {
        onUploadError(`Connection error uploading ${file.name}. Is the backend running on port 8000?`);
      }
    }
    setUploadProgress('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      await uploadFiles(files);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  // Status variables
  const isHealthy = status.status === 'healthy';
  const isBusy = status.status === 'busy' || status.status === 'starting';
  const dbType = status.active_db || 'offline';
  
  const getDbBadgeDetails = () => {
    if (dbType === 'chromadb') {
      return { label: 'ChromaDB', color: '#00f2fe' };
    } else if (dbType === 'sqlite') {
      return { label: 'SQLite Fallback Engine', color: '#fbbf24' };
    } else if (dbType === 'starting' || status.status === 'starting') {
      return { label: 'ChromaDB (Connecting...)', color: '#38bdf8' };
    }
    return { label: 'Disconnected', color: '#ef4444' };
  };

  const dbBadge = getDbBadgeDetails();

  return (
    <div className="sidebar">
      {/* Sidebar Header */}
      <div className="sidebar-header">
        <h2>🎓 EduQuery AI</h2>
      </div>

      {/* System Status Indicators */}
      <div className="status-card">
        <div className="status-row">
          <span className="status-label">System Status:</span>
          <span className="status-value" style={{ display: 'flex', alignItems: 'center', gap: '4px', color: isHealthy ? '#34d399' : isBusy ? '#38bdf8' : '#f87171' }}>
            <Activity size={12} className={isHealthy || isBusy ? 'pulse' : ''} />
            {isHealthy ? 'Connected' : isBusy ? 'Connecting...' : 'Offline'}
          </span>
        </div>
        <div className="status-row">
          <span className="status-label">Vector DB:</span>
          <span className="status-value" style={{ color: dbBadge.color }}>
            {dbBadge.label}
          </span>
        </div>
      </div>

      {/* Ingestion Dropzone */}
      <div className="upload-container">
        <div className="upload-title">📥 Upload Coursework</div>
        <div
          className="drag-drop-zone"
          style={{
            borderColor: isDragging ? '#00f2fe' : 'rgba(79, 172, 254, 0.4)',
            background: isDragging ? 'rgba(0, 242, 254, 0.06)' : 'rgba(15, 23, 42, 0.4)',
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={triggerFileInput}
        >
          <UploadCloud size={32} style={{ color: isDragging ? '#00f2fe' : '#4facfe' }} />
          <p style={{ fontWeight: '500', fontSize: '0.85rem' }}>
            {uploadProgress || 'Drag & drop lectures/notes here'}
          </p>
          <p style={{ fontSize: '0.75rem', opacity: 0.7 }}>
            Supports PDF, PPTX, PPT, TXT, MD
          </p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            onClick={(e) => e.stopPropagation()}
            multiple
            className="file-input-hidden"
            accept=".pdf,.pptx,.ppt,.txt,.md"
          />
        </div>
      </div>

      {/* Document library manager list */}
      <div className="library-container">
        <div className="library-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>📚 Study Library</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              ({documents.length} files)
            </span>
            <button
              type="button"
              className="doc-delete-btn"
              onClick={onRefreshLibrary}
              title="Refresh library from backend"
            >
              <RefreshCw size={14} />
            </button>
          </span>
        </div>
        
        <div className="library-list">
          {documents.length === 0 ? (
            <div className="no-docs">
              No coursework uploaded yet. Upload documents above to begin learning!
            </div>
          ) : (
            documents.map((doc, idx) => {
              const docType = doc.type?.toUpperCase();
              let docMeta = docType;
              if (docType === 'PDF' && doc.total_pages) {
                docMeta = `PDF • ${doc.total_pages} pages`;
              } else if ((docType === 'PPTX' || docType === 'PPT') && (doc.slide_count || doc.total_pages)) {
                docMeta = `PPTX • ${doc.slide_count || doc.total_pages} slides`;
              }
              if (doc.indexed === true) {
                docMeta += ' • Indexed';
              } else if (doc.indexed === false) {
                docMeta += ' • Pending indexing';
              }

              return (
                <div key={idx} className="doc-item">
                  <div className="doc-info">
                    {getDocIcon(doc.type)}
                    <div className="doc-details">
                      <span className="doc-name" title={doc.source}>
                        {doc.source}
                      </span>
                      <span className="doc-meta">{docMeta}</span>
                    </div>
                  </div>
                  <button
                    className="doc-delete-btn"
                    onClick={() => onDeleteDoc(doc.source)}
                    title={`Remove ${doc.source}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

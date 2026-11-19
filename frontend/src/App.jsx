import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, GraduationCap } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatThread from './components/ChatThread';
import QuickPrompts from './components/QuickPrompts';
import { apiFetch, normalizeDocuments, parseApiError, readJson } from './api';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const libraryRequestId = useRef(0);
  const [status, setStatus] = useState({ active_db: 'Loading...', status: 'offline', document_count: 0 });
  const [isQuerying, setIsQuerying] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [toastMessage, setToastMessage] = useState('');
  const [toastTone, setToastTone] = useState('success');

  const showToast = (msg, tone = 'success') => {
    setToastTone(tone);
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage('');
    }, 4000);
  };

  const fetchLibrary = useCallback(async () => {
    const requestId = ++libraryRequestId.current;
    try {
      const response = await apiFetch('/api/documents');
      const data = await readJson(response);
      if (!response.ok) {
        throw new Error(parseApiError(data, 'Failed to load document library'));
      }
      const docs = normalizeDocuments(data);
      if (requestId === libraryRequestId.current) {
        setDocuments(docs);
      }
      return docs;
    } catch (e) {
      console.error('Error loading document library:', e);
      return null;
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await apiFetch('/api/status');
      const data = await readJson(response);
      if (response.ok) {
        setStatus(data);
      } else {
        setStatus({ active_db: 'None', status: 'offline', document_count: 0 });
      }
    } catch (e) {
      setStatus({ active_db: 'None', status: 'offline', document_count: 0 });
    }
  }, []);

  const refreshWorkspace = useCallback(async () => {
    await fetchLibrary();
    await fetchStatus();
  }, [fetchLibrary, fetchStatus]);

  useEffect(() => {
    refreshWorkspace();
    const interval = setInterval(refreshWorkspace, 10000);
    return () => clearInterval(interval);
  }, [refreshWorkspace]);

  const handleDeleteDoc = async (filename) => {
    try {
      const response = await apiFetch(`/api/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      });
      const data = await readJson(response);
      if (response.ok) {
        showToast(`Successfully removed ${filename}`);
        setDocuments((current) => current.filter((doc) => doc.source !== filename));
        await refreshWorkspace();
      } else {
        showToast(parseApiError(data, `Failed to delete ${filename}`), 'error');
      }
    } catch (err) {
      showToast('Connection error during deletion', 'error');
    }
  };

  const handleSendMessage = async (textToSend) => {
    const text = (textToSend ?? inputMessage).trim();
    if (!text || isQuerying) return;

    setInputMessage('');
    setIsQuerying(true);

    const userMsg = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const response = await apiFetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, n_results: 5 }),
      });

      const data = await readJson(response);
      if (response.ok) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: data.answer,
            sources: data.sources || [],
            metrics: data.metrics || null,
          },
        ]);
      } else {
        const errMsg = parseApiError(data, 'Server side parsing error.');
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `⚠️ **Server Connection Issue**\n\n${errMsg}`,
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            '⚠️ **Network Connection Failed**\n\nCould not reach the backend tutor server. Start FastAPI on port 8000 (`python -m uvicorn backend.app:app --reload --port 8000`) and keep the Vite app on port 5173.',
        },
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleSelectPrompt = (promptText) => {
    handleSendMessage(promptText);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const hasDocuments = documents.length > 0 || (status.document_count || 0) > 0;

  return (
    <div className="app-container">
      {toastMessage && (
        <div className={`toast-msg ${toastTone === 'error' ? 'toast-error' : ''}`}>{toastMessage}</div>
      )}

      <Sidebar
        documents={documents}
        status={status}
        onDeleteDoc={handleDeleteDoc}
        onRefreshLibrary={refreshWorkspace}
        onUploadStart={() => showToast('Indexing document...')}
        onUploadSuccess={async (msg, uploadedDocument) => {
          showToast(msg);
          if (uploadedDocument) {
            setDocuments((currentDocuments) => [
              uploadedDocument,
              ...currentDocuments.filter((doc) => doc.source !== uploadedDocument.source),
            ]);
          }
          for (let attempt = 0; attempt < 6; attempt += 1) {
            const docs = await fetchLibrary();
            if (docs && (!uploadedDocument || docs.some((doc) => doc.source === uploadedDocument.source))) {
              break;
            }
            await new Promise((resolve) => setTimeout(resolve, 500));
          }
          await fetchStatus();
        }}
        onUploadError={(err) => showToast(err, 'error')}
      />

      <div className="main-content">
        <div className="classroom-header">
          <h1>🎓 Classroom AI Workspace</h1>
          <p>Clarify units, generate quizzes, and summarize chapters based strictly on your documents.</p>
        </div>

        <div className="chat-panel">
          <div className="chat-title-bar">
            <GraduationCap size={20} style={{ color: 'var(--accent-cyan)' }} />
            <span>Classroom Discussion Board</span>
          </div>

          <ChatThread messages={messages} isLoading={isQuerying} />

          <QuickPrompts onSelectPrompt={handleSelectPrompt} disabled={isQuerying} />

          <form
            className="input-panel"
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
          >
            <div className="input-container">
              <textarea
                className="chat-input"
                rows={1}
                placeholder={
                  hasDocuments
                    ? 'Ask a question about your study materials...'
                    : 'Upload a document, then ask a question about your study materials...'
                }
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isQuerying}
              />
            </div>
            <button
              type="submit"
              className="send-btn"
              disabled={isQuerying || !inputMessage.trim()}
              aria-label="Send message"
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

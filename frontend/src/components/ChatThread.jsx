import React, { useEffect, useRef } from 'react';
import { BookOpen, Presentation, FileText } from 'lucide-react';

export default function ChatThread({ messages, isLoading }) {
  const bottomRef = useRef(null);

  // Auto-scroll to the bottom of the chat when messages update or loading starts
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const getSourceIcon = (type) => {
    switch (type?.toLowerCase()) {
      case 'pdf':
        return <BookOpen size={13} />;
      case 'pptx':
      case 'ppt':
        return <Presentation size={13} />;
      default:
        return <FileText size={13} />;
    }
  };

  return (
    <div className="chat-history">
      {messages.length === 0 ? (
        <div className="chat-empty-state">
          🎒 Classroom chat initialized. Ask a question about your study materials, or select a prompt suggestion chip below to begin.
        </div>
      ) : (
        messages.map((msg, idx) => (
          <div key={idx} className={`msg-row ${msg.role}`}>
            <div className="msg-bubble">
              <div style={{ whiteSpace: 'pre-line' }}>{msg.content}</div>
              
              {/* Display Benchmarking Telemetry */}
              {msg.role === 'assistant' && msg.metrics && (
                <div className="telemetry-bar">
                  <span className="telemetry-item">
                    ⚡ Latency: {msg.metrics.latency_seconds}s
                  </span>
                  <span className="telemetry-divider">|</span>
                  <span className="telemetry-item">
                    🚀 Speed: {msg.metrics.tokens_per_second} t/s
                  </span>
                  <span className="telemetry-divider">|</span>
                  <span className="telemetry-item">
                    📝 {msg.metrics.tokens_generated} tokens
                  </span>
                  <span className="telemetry-divider">|</span>
                  <span className="telemetry-item" style={{ textTransform: 'capitalize' }}>
                    🤖 via {msg.metrics.provider}
                  </span>
                </div>
              )}
              
              {/* Display Sources shelf if assistant response has citations */}
              {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                <div style={{ marginTop: '14px' }}>
                  <div className="sources-title">
                    <span>📖 Citations & Context Sources:</span>
                  </div>
                  <div className="sources-container">
                    {msg.sources.map((src, sIdx) => {
                      const tooltip = `${src.document} (${src.reference})${src.title ? ` - ${src.title}` : ''}`;
                      return (
                        <div
                          key={sIdx}
                          className="source-badge"
                          title={tooltip}
                        >
                          {getSourceIcon(src.type)}
                          <span>
                            {src.document} • {src.reference}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))
      )}

      {/* Pulsing loading indicator */}
      {isLoading && (
        <div className="msg-row assistant">
          <div className="msg-bubble" style={{ background: 'rgba(30, 41, 59, 0.25)', padding: '10px 16px' }}>
            <div className="loader-row">
              <span className="pulse-dot"></span>
              <span className="pulse-dot"></span>
              <span className="pulse-dot"></span>
              <span>Tutor is formulating answer...</span>
            </div>
          </div>
        </div>
      )}
      
      <div ref={bottomRef} />
    </div>
  );
}
